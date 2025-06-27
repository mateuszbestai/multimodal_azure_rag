import os
import json
import time
import logging
import re
import fitz  # PyMuPDF
from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict, List
from dotenv import load_dotenv
import nest_asyncio
import io
from PIL import Image

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.formrecognizer import DocumentAnalysisClient

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType
)

# Async imports
import asyncio
from azure.storage.blob.aio import BlobServiceClient
import aiofiles

from excel_ingest import ExcelProcessor

nest_asyncio.apply()
load_dotenv()

# ================== Configuration ==================
# Set Azure OpenAI environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"  # Updated to stable version

# Environment Variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "rag-demo-images"
INDEX_NAME = "azure-multimodal-search-new1"
DOWNLOAD_PATH = "pdf-files"

# Optimization settings
IMAGE_DPI = 100  # Reduced from 150 for faster processing
IMAGE_FORMAT = "JPEG"  # Changed from PNG for smaller files
IMAGE_QUALITY = 85  # JPEG quality (1-100)
MAX_CONCURRENT_UPLOADS = 15  # Increased from 5

# Initialize Azure OpenAI settings first
Settings.llm = AzureOpenAI(
    engine=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",  # Match API version
    api_type="azure"
)

Settings.embed_model = AzureOpenAIEmbedding(
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,  # Correct parameter
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",  # Correct version for embeddings
    api_type="azure"
)

# Initialize Azure Clients
document_analysis_client = DocumentAnalysisClient(
    endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY),
)

# Initialize Search Clients
search_credential = AzureKeyCredential(SEARCH_SERVICE_API_KEY)
index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=search_credential)
search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=search_credential)

# Define metadata fields for the index
metadata_fields = {
    "page_num": ("page_num", MetadataIndexFieldType.INT64),
    "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
    "image_path": ("image_path", MetadataIndexFieldType.STRING),
    "full_text": ("full_text", MetadataIndexFieldType.STRING),
}

SUPPORTED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.xlsm'}

def get_file_type(file_path: str) -> str:
    """Determine file type from extension."""
    extension = Path(file_path).suffix.lower()
    if extension == '.pdf':
        return 'pdf'
    elif extension in {'.xlsx', '.xls', '.xlsm'}:
        return 'excel'
    else:
        raise ValueError(f"Unsupported file type: {extension}")

# ================== Document Processing ==================
def create_folder_structure(base_path: str, pdf_path: str) -> str:
    """Create organized folder structure for output files."""
    folder_name = Path(pdf_path).stem
    folder_path = Path(base_path) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return str(folder_path)

def pdf_to_images_optimized(pdf_path: str, output_base: str) -> List[dict]:
    """Convert PDF to optimized images with lower DPI and JPEG format."""
    image_dicts = []
    folder_path = create_folder_structure(output_base, pdf_path)
    
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise ValueError("Encrypted PDF - password required")

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                # Lower DPI for smaller files
                pix = page.get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csRGB, alpha=False)
                
                # Save as JPEG instead of PNG
                image_name = f"page_{page_num+1}.jpg"
                image_path = str(Path(folder_path) / image_name)
                
                # Convert to PIL Image for JPEG saving with quality control
                img_data = pix.pil_tobytes(format="JPEG", optimize=True)
                img = Image.open(io.BytesIO(img_data))
                img.save(image_path, "JPEG", quality=IMAGE_QUALITY, optimize=True)
                
                image_dicts.append({
                    "name": image_name,
                    "path": image_path,
                    "page_num": page_num + 1
                })
                
            except Exception as e:
                logging.error(f"Page {page_num+1} processing failed: {str(e)}", exc_info=True)
                continue
                
        return image_dicts

    except Exception as e:
        logging.error(f"PDF processing failed: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def extract_document_data(pdf_path: str) -> dict:
    """Extract text and images from PDF."""
    with open(pdf_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()

    # Use optimized image conversion
    image_dicts = pdf_to_images_optimized(pdf_path, DOWNLOAD_PATH)

    return {
        "text_content": result.content,
        "pages": [{"text": "\n".join(line.content for line in page.lines)} for page in result.pages],
        "images": image_dicts,
        "source_path": pdf_path
    }

# ================== Optimized Azure Blob Storage ==================
class OptimizedBlobUploader:
    """Optimized blob uploader with connection reuse and better concurrency."""
    
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = None
        
    async def __aenter__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        # Ensure container exists
        container_client = self.blob_service_client.get_container_client(self.container_name)
        if not await container_client.exists():
            await container_client.create_container()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.blob_service_client:
            await self.blob_service_client.close()
    
    async def upload_images_batch(self, image_dicts: List[dict]) -> Dict[str, str]:
        """Upload images in batch with high concurrency."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
        tasks = []
        
        for img in image_dicts:
            task = self._upload_single_optimized(img, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        uploaded_urls = {}
        for img, result in zip(image_dicts, results):
            if isinstance(result, Exception):
                logging.error(f"Upload failed for {img['name']}: {str(result)}")
            elif result:
                uploaded_urls[img["name"]] = result
                
        return uploaded_urls
    
    async def _upload_single_optimized(self, image: dict, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Upload single image with optimization."""
        async with semaphore:
            try:
                blob_name = image["name"]
                
                # Read file asynchronously
                async with aiofiles.open(image["path"], "rb") as f:
                    image_data = await f.read()
                
                # Get blob client
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                # Upload with optimized settings
                await blob_client.upload_blob(
                    image_data, 
                    overwrite=True,
                    max_concurrency=4,  # Parallel chunk upload
                    length=len(image_data)
                )
                
                return blob_client.url
                
            except Exception as e:
                logging.error(f"Upload failed for {image['name']}: {str(e)}")
                return None

async def upload_images_concurrently(image_dicts: List[dict]) -> Dict[str, str]:
    """Upload images using optimized uploader."""
    async with OptimizedBlobUploader(BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME) as uploader:
        return await uploader.upload_images_batch(image_dicts)

# ================== Search Index Integration ==================
def create_search_nodes(document_data: dict, image_urls: Dict[str, str]) -> List[TextNode]:
    """Create search nodes with linked text and images."""
    nodes = []
    page_image_map = {
        int(re.search(r"page_(\d+)", name).group(1)): url
        for name, url in image_urls.items()
    }

    for page_num, page_text in enumerate(document_data["pages"], start=1):
        image_url = page_image_map.get(page_num)
        if not image_url:
            logging.warning(f"No image found for page {page_num}")
            continue

        node = TextNode(
            text=page_text["text"],
            metadata={
                "page_num": page_num,
                "image_path": image_url,
                "doc_id": Path(document_data["source_path"]).stem,
                "full_text": page_text["text"]
            }
        )
        nodes.append(node)
    
    return nodes

def create_vector_store(
    index_client,  # now using the SearchIndexClient
    use_existing_index: bool = False
) -> AzureAISearchVectorStore:
    """Create or get existing Azure AI Search vector store."""
    # IMPORTANT: Must specify index_name if passing SearchIndexClient
    return AzureAISearchVectorStore(
        search_or_index_client=index_client,
        index_name=INDEX_NAME,  # <-- The fix
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="id",
        chunk_field_key="full_text",
        embedding_field_key="embedding",
        embedding_dimensionality=3072,
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        filterable_metadata_field_keys=metadata_fields,
        language_analyzer="en.lucene",
        vector_algorithm_type="exhaustiveKnn",
    )

def create_or_load_index(
    text_nodes,
    index_client,  # now expect the index_client
    embed_model,
    llm,
    use_existing_index: bool = False
) -> VectorStoreIndex:
    """Create new index or load existing one."""
    vector_store = create_vector_store(index_client, use_existing_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if use_existing_index:
        return VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
        )
    else:
        return VectorStoreIndex(
            nodes=text_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
            show_progress=True,
        )

async def process_document(file_path: str) -> RetrieverQueryEngine:
    """Enhanced document processing pipeline supporting both PDF and Excel files."""
    try:
        file_type = get_file_type(file_path)
        
        if file_type == 'pdf':
            return await process_pdf_document(file_path)
        elif file_type == 'excel':
            return await process_excel_document(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

# ================== Main Processing Pipeline ==================
async def process_pdf_document(pdf_path: str) -> RetrieverQueryEngine:
    """Process PDF documents with optimized image handling."""
    start_time = time.time()
    
    # Extract document data
    document_data = extract_document_data(pdf_path)
    extract_time = time.time() - start_time
    logging.info(f"Document extraction took {extract_time:.2f}s")
    
    # Upload images with optimized settings
    upload_start = time.time()
    image_urls = await upload_images_concurrently(document_data["images"])
    upload_time = time.time() - upload_start
    logging.info(f"Uploaded {len(image_urls)} images in {upload_time:.2f}s")
    
    # Create search nodes
    nodes = create_search_nodes(document_data, image_urls)
    
    # Create index
    index_start = time.time()
    index = create_or_load_index(
        text_nodes=nodes,
        index_client=index_client,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        use_existing_index=False
    )
    index_time = time.time() - index_start
    logging.info(f"Index creation took {index_time:.2f}s")

    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        response_mode="compact"
    )
    
    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f}s")
    
    return RetrieverQueryEngine(
        retriever=index.as_retriever(similarity_top_k=3),
        response_synthesizer=response_synthesizer
    )

async def process_excel_document(excel_path: str) -> RetrieverQueryEngine:
    """Process Excel documents using the new ExcelProcessor."""
    try:
        # Initialize Excel processor with optimized settings
        excel_processor = ExcelProcessor(
            blob_connection_string=BLOB_CONNECTION_STRING,
            container_name=BLOB_CONTAINER_NAME
        )
        
        # Process Excel file
        logging.info(f"Processing Excel file: {excel_path}")
        excel_data = await excel_processor.process_excel_file(excel_path)
        
        # Upload any extracted images with optimization
        image_urls = {}
        if excel_data.get("images"):
            async with OptimizedBlobUploader(BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME) as uploader:
                image_urls = await uploader.upload_images_batch(excel_data["images"])
            logging.info(f"Uploaded {len(image_urls)} Excel images")
        
        # Create search nodes
        nodes = excel_processor.create_search_nodes(excel_data, image_urls)
        logging.info(f"Created {len(nodes)} search nodes from Excel file")
        
        # Create or load index
        index = create_or_load_index(
            text_nodes=nodes,
            index_client=index_client,
            embed_model=Settings.embed_model,
            llm=Settings.llm,
            use_existing_index=False
        )

        response_synthesizer = get_response_synthesizer(
            llm=Settings.llm,
            response_mode="compact"
        )
        
        return RetrieverQueryEngine(
            retriever=index.as_retriever(similarity_top_k=5),  # Increased for Excel data
            response_synthesizer=response_synthesizer
        )
        
    except Exception as e:
        logging.error(f"Excel processing failed: {str(e)}")
        raise

async def process_multiple_files(file_paths: List[str]) -> Dict[str, RetrieverQueryEngine]:
    """Process multiple files and return individual query engines."""
    query_engines = {}
    
    for file_path in file_paths:
        try:
            file_name = Path(file_path).name
            logging.info(f"Processing file: {file_name}")
            
            query_engine = await process_document(file_path)
            query_engines[file_name] = query_engine
            
            logging.info(f"Successfully processed: {file_name}")
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {str(e)}")
            continue
    
    return query_engines

async def process_directory(directory_path: str) -> Dict[str, RetrieverQueryEngine]:
    """Process all supported files in a directory."""
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all supported files
    file_paths = []
    for extension in SUPPORTED_EXTENSIONS:
        file_paths.extend(directory.glob(f"*{extension}"))
    
    if not file_paths:
        logging.warning(f"No supported files found in {directory_path}")
        return {}
    
    logging.info(f"Found {len(file_paths)} files to process")
    return await process_multiple_files([str(p) for p in file_paths])

# ================== Enhanced CLI Interface ==================
def main():
    """Enhanced main function with file type detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents (PDF or Excel) for RAG system")
    parser.add_argument("path", help="Path to file or directory to process")
    parser.add_argument("--file-type", choices=['pdf', 'excel', 'auto'], default='auto',
                      help="Specify file type (auto-detect by default)")
    parser.add_argument("--batch", action='store_true',
                      help="Process entire directory")
    parser.add_argument("--output-dir", default="output",
                      help="Output directory for processed files")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('document_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Log optimization settings
    logging.info(f"Starting with optimized settings:")
    logging.info(f"  - Image DPI: {IMAGE_DPI}")
    logging.info(f"  - Image Format: {IMAGE_FORMAT}")
    logging.info(f"  - Image Quality: {IMAGE_QUALITY}")
    logging.info(f"  - Max Concurrent Uploads: {MAX_CONCURRENT_UPLOADS}")
    
    try:
        if args.batch:
            # Process entire directory
            query_engines = asyncio.run(process_directory(args.path))
            logging.info(f"Successfully processed {len(query_engines)} files")
            
            # Save results summary
            summary = {
                "processed_files": list(query_engines.keys()),
                "total_count": len(query_engines),
                "timestamp": time.time(),
                "optimization_settings": {
                    "image_dpi": IMAGE_DPI,
                    "image_format": IMAGE_FORMAT,
                    "image_quality": IMAGE_QUALITY,
                    "max_concurrent_uploads": MAX_CONCURRENT_UPLOADS
                }
            }
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "processing_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
        else:
            # Process single file
            if args.file_type == 'auto':
                file_type = get_file_type(args.path)
            else:
                file_type = args.file_type
            
            logging.info(f"Processing {file_type} file: {args.path}")
            query_engine = asyncio.run(process_document(args.path))
            logging.info("Processing completed successfully")
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())