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
INDEX_NAME = "azure-multimodal-search"
DOWNLOAD_PATH = "pdf-files"

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

# ================== Document Processing ==================
def create_folder_structure(base_path: str, pdf_path: str) -> str:
    """Create organized folder structure for output files."""
    folder_name = Path(pdf_path).stem
    folder_path = Path(base_path) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return str(folder_path)

def pdf_to_images(pdf_path: str, output_base: str) -> List[dict]:
    """Convert PDF to images with enhanced error handling."""
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
                pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB, alpha=False)
                image_name = f"page_{page_num+1}.png"
                image_path = str(Path(folder_path) / image_name)
                
                pix.save(image_path)
                
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

    image_dicts = pdf_to_images(pdf_path, DOWNLOAD_PATH)

    return {
        "text_content": result.content,
        "pages": [{"text": "\n".join(line.content for line in page.lines)} for page in result.pages],
        "images": image_dicts,
        "source_path": pdf_path
    }

# ================== Azure Blob Storage ==================
async def create_container_if_not_exists():
    """Ensure the blob container exists."""
    async with BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING) as blob_service_client:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        if not await container_client.exists():
            await container_client.create_container()

async def upload_images_concurrently(image_dicts: List[dict]) -> Dict[str, str]:
    """Upload images with concurrency control."""
    await create_container_if_not_exists()
    
    semaphore = asyncio.Semaphore(5)
    tasks = [upload_single_image(img, semaphore) for img in image_dicts]
    results = await asyncio.gather(*tasks)
    
    return {img["name"]: url for img, url in zip(image_dicts, results) if url}

async def upload_single_image(image: dict, semaphore: asyncio.Semaphore) -> Optional[str]:
    """Upload individual image with error handling."""
    async with semaphore:
        try:
            async with BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING) as service_client:
                container_client = service_client.get_container_client(BLOB_CONTAINER_NAME)
                blob_client = container_client.get_blob_client(image["name"])
                with open(image["path"], "rb") as data:
                    await blob_client.upload_blob(data, overwrite=True)
                    return blob_client.url
        except Exception as e:
            logging.error(f"Upload failed for {image['name']}: {str(e)}")
            return None

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
        embedding_dimensionality=1536,
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

# ================== Main Processing Pipeline ==================
async def process_document(pdf_path: str) -> RetrieverQueryEngine:
    """End-to-end document processing pipeline."""
    try:
        document_data = extract_document_data(pdf_path)
        image_urls = await upload_images_concurrently(document_data["images"])
        logging.info(f"Uploaded {len(image_urls)} images")
        
        nodes = create_search_nodes(document_data, image_urls)
        
        # Pass the SearchIndexClient instead of SearchClient.
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
            retriever=index.as_retriever(similarity_top_k=3),
            response_synthesizer=response_synthesizer
        )

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pdf_path = "data/pdfs/new-relic-2024-observability-forecast-report.pdf"
    
    try:
        query_engine = asyncio.run(process_document(pdf_path))
        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")