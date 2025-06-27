from datetime import datetime
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from typing import Optional, List
import logging
import requests
import tempfile
import shutil
import asyncio
from pathlib import Path
from werkzeug.utils import secure_filename
import json
import tiktoken

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode, TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType
)

# Import the Excel processor
from excel_ingest import ExcelProcessor

# ================== Load Environment ==================
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FrontendConfig:
    """Centralized configuration for frontend components"""
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    SEARCH_SERVICE_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    INDEX_NAME = "azure-multimodal-search-new1"
    BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "rag-demo-images")
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
    
    # File upload settings
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'xlsm', 'pdf'}

    @classmethod
    def build_image_url(cls, blob_path: str) -> str:
        """Match Chainlit's direct URL handling with SAS validation"""
        if not blob_path:
            return ""
        
        if blob_path.startswith("http"):
            return blob_path
        
        clean_path = blob_path.lstrip('/')
        encoded_path = requests.utils.quote(clean_path)
        
        sas = cls.SAS_TOKEN
        if sas and not sas.startswith('?'):
            sas = f'?{sas}'
        
        return (
            f"https://{cls.STORAGE_ACCOUNT_NAME}.blob.core.windows.net/"
            f"{cls.BLOB_CONTAINER}/{encoded_path}"
            f"{sas}"
        )

def validate_env():
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_SEARCH_SERVICE_ENDPOINT',
        'AZURE_SEARCH_ADMIN_KEY',
        'AZURE_STORAGE_ACCOUNT_NAME',
        'AZURE_STORAGE_SAS_TOKEN',
        'BLOB_CONNECTION_STRING'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

validate_env()

# Log configuration for debugging
logger.info(f"Azure OpenAI Endpoint: {FrontendConfig.AZURE_OPENAI_ENDPOINT}")
logger.info(f"Chat Deployment Name: {FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT}")

# ================== Initialize Flask ==================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = FrontendConfig.MAX_CONTENT_LENGTH
CORS(app)

# ================== Embedding Model Detection ==================
def get_embedding_dimensions(model_name: str) -> int:
    """Get the correct dimensions for the embedding model."""
    model_dimensions = {
        'text-embedding-ada-002': 1536,
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
    }
    
    # Default to ada-002 if model not found
    return model_dimensions.get(model_name, 1536)

def detect_current_embedding_model():
    """Detect which embedding model is being used and its dimensions."""
    model_name = FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    embedding_dim = get_embedding_dimensions(model_name)
    logger.info(f"Detected embedding model: {model_name} with {embedding_dim} dimensions")
    return model_name, embedding_dim

# ================== Text Chunking Utilities ==================
def count_tokens(text: str, model_name: str = "text-embedding-3-large") -> int:
    """Count tokens in text for the given model."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Could not count tokens: {e}")
        return len(text.split()) * 1.3

def chunk_text(text: str, max_tokens: int = 7000, overlap: int = 200) -> List[str]:
    """Split text into chunks that fit within token limits."""
    if count_tokens(text) <= max_tokens:
        return [text]
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        
        if count_tokens(potential_chunk) <= max_tokens:
            current_chunk = potential_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk + ".")
                overlap_text = ". ".join(sentences[-2:]) if len(sentences) > 2 else ""
                current_chunk = overlap_text + ". " + sentence if overlap_text else sentence
            else:
                words = sentence.split()
                word_chunks = []
                current_word_chunk = ""
                
                for word in words:
                    potential_word_chunk = current_word_chunk + " " + word if current_word_chunk else word
                    if count_tokens(potential_word_chunk) <= max_tokens:
                        current_word_chunk = potential_word_chunk
                    else:
                        if current_word_chunk:
                            word_chunks.append(current_word_chunk)
                        current_word_chunk = word
                
                if current_word_chunk:
                    word_chunks.append(current_word_chunk)
                
                chunks.extend(word_chunks)
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk + ".")
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# ================== Multimodal LLM ==================
# FIX: Updated to use correct parameters and API version
try:
    azure_openai_mm_llm = AzureOpenAIMultiModal(
        deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,  # Changed from engine
        api_version="2024-02-15-preview",  # Updated for multimodal support
        model="gpt-4.1",  # Specify vision model
        max_new_tokens=4096,
        api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
        azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,  # Changed from api_base
    )
    logger.info("Initialized multimodal LLM successfully")
except Exception as e:
    logger.warning(f"Failed to initialize multimodal LLM: {e}")
    logger.warning("Will use text-only fallback for queries")
    azure_openai_mm_llm = None

# ================== Enhanced Prompt Template ==================
QA_PROMPT_TMPL = """\
You are a helpful AI assistant with access to both text and images. 
Use the document text and any associated images to provide the best possible answer.
Do not use knowledge outside of the provided documents.

DOCUMENT CONTEXT:
{context_str}

INSTRUCTIONS:
1. If using image information, clearly state which page(s) you are referencing.
2. Integrate text and image details to form a coherent answer.
3. If there are contradictions or missing information, explain them.
4. Give a concise yet thorough answer, and cite relevant pages or images.

USER QUERY:
{query_str}

Now craft your final answer:
"""
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

# ================== Enhanced Query Engine ==================
class VisionQueryEngine(CustomQueryEngine):
    """Updated query engine with fallback support"""
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: Optional[AzureOpenAIMultiModal]

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs):
        # Make multi_modal_llm optional
        if 'multi_modal_llm' not in kwargs:
            kwargs['multi_modal_llm'] = None
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str) -> Response:
        nodes = self.retriever.retrieve(query_str)
        
        # Build image nodes with page number references
        image_nodes = []
        for n in nodes:
            blob_path = n.metadata.get("image_path")
            if blob_path:
                try:
                    full_url = FrontendConfig.build_image_url(blob_path)
                    img_node = ImageNode(image_url=full_url)
                    img_node.metadata = {"page_num": n.metadata.get("page_num", "N/A")}
                    image_nodes.append(NodeWithScore(node=img_node))
                except Exception as e:
                    logger.error(f"Image node error: {str(e)}")
        
        # Build the textual context (include page numbers)
        context_str = "\n".join([
            f"Page {n.metadata.get('page_num', '?')}: {n.get_content(metadata_mode=MetadataMode.LLM)}"
            for n in nodes
        ])
        
        formatted_prompt = self.qa_prompt.format(
            context_str=context_str,
            query_str=query_str
        )
        
        # Try multimodal first if available
        if self.multi_modal_llm and image_nodes:
            try:
                logger.debug(f"Using multimodal LLM with {len(image_nodes)} images")
                response = self.multi_modal_llm.complete(
                    prompt=formatted_prompt,
                    image_documents=[n.node for n in image_nodes],
                )
                
                if response and str(response).strip():
                    logger.debug("Multimodal response successful")
                else:
                    raise ValueError("Empty response from multimodal LLM")
                    
            except Exception as e:
                logger.warning(f"Multimodal LLM failed: {str(e)}")
                logger.info("Falling back to text-only mode")
                # Fall through to text-only mode
                response = None
        else:
            response = None
        
        # Use text-only mode if multimodal failed or unavailable
        if response is None:
            try:
                logger.debug("Using text-only LLM")
                response = Settings.llm.complete(formatted_prompt)
                
                if not response or not str(response).strip():
                    raise ValueError("Empty response from text LLM")
                    
            except Exception as e:
                logger.error(f"Text LLM also failed: {str(e)}")
                raise

        # Build references
        references = []
        for n in nodes:
            ref_text = f"Page {n.metadata.get('page_num', 'N/A')}: {n.get_content(metadata_mode=MetadataMode.LLM)[:100]}..."
            if n.metadata.get("image_path"):
                ref_text += " [Image available]"
            references.append(ref_text)
        
        return Response(
            response=str(response),
            source_nodes=nodes,
            metadata={
                "references": references,
                "pages": list({int(n.metadata.get("page_num", 0)) for n in nodes if n.metadata.get("page_num")}),
                "images": image_nodes if self.multi_modal_llm else []
            }
        )

# ================== Initialize Query Engine ==================
def get_available_fields():
    """Get available fields from the current index schema."""
    try:
        credential = AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        index_client = SearchIndexClient(endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT, credential=credential)
        existing_index = index_client.get_index(FrontendConfig.INDEX_NAME)
        return {field.name for field in existing_index.fields}
    except Exception as e:
        logger.warning(f"Could not get index fields: {e}")
        return {"id", "full_text", "metadata", "doc_id", "embedding"}

def get_index_embedding_dimensions():
    """Get the embedding dimensions configured in the existing index."""
    try:
        credential = AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        index_client = SearchIndexClient(endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT, credential=credential)
        existing_index = index_client.get_index(FrontendConfig.INDEX_NAME)
        
        # Find the embedding field and get its dimensions
        for field in existing_index.fields:
            if field.name == "embedding":
                if hasattr(field, 'vector_search_dimensions'):
                    return field.vector_search_dimensions
                break
        
        # Default to 1536 if not found
        return 1536
        
    except Exception as e:
        logger.warning(f"Could not get index embedding dimensions: {e}")
        return 1536

def initialize_engine():
    """Initialize Azure components and query engine"""
    try:
        # Detect embedding model and dimensions
        embedding_model, model_dimensions = detect_current_embedding_model()
        index_dimensions = get_index_embedding_dimensions()
        
        logger.info(f"Model dimensions: {model_dimensions}, Index dimensions: {index_dimensions}")
        
        # Check for dimension mismatch
        if model_dimensions != index_dimensions:
            logger.warning(f"Dimension mismatch detected!")
            logger.warning(f"Embedding model '{embedding_model}' produces {model_dimensions}D vectors")
            logger.warning(f"But your index expects {index_dimensions}D vectors")
            logger.warning(f"Reverting to text-embedding-ada-002 to match your index")
            
            # Force use of ada-002 model to match index
            actual_embedding_model = "text-embedding-3-large"
            actual_dimensions = 3072
        else:
            actual_embedding_model = embedding_model
            actual_dimensions = model_dimensions

        # FIX: Updated LLM initialization with correct parameters
        llm = AzureOpenAI(
            model="gpt-4.1",  # Explicit model
            deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",  # Updated API version
            streaming=True
        )

        # Use compatible embedding model
        embed_model = AzureOpenAIEmbedding(
            model=actual_embedding_model,
            deployment_name=FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2023-05-15",
        )

        Settings.llm = llm
        Settings.embed_model = embed_model

        # Get available fields from the index
        available_fields = get_available_fields()
        logger.info(f"Available index fields: {available_fields}")

        # Only use fields that definitely exist in the current index
        safe_filterable_fields = {}
        
        # Core fields that should always exist
        core_fields = {
            "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
            "full_text": ("full_text", MetadataIndexFieldType.STRING),
        }
        
        # Optional fields that might exist
        optional_fields = {
            "page_num": ("page_num", MetadataIndexFieldType.INT64),
            "image_path": ("image_path", MetadataIndexFieldType.STRING),
        }
        
        # Add fields only if they exist
        for field_name, field_config in {**core_fields, **optional_fields}.items():
            if field_name in available_fields:
                safe_filterable_fields[field_name] = field_config

        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )

        vector_store = AzureAISearchVectorStore(
            search_or_index_client=search_client,
            id_field_key="id",
            chunk_field_key="full_text",
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            embedding_field_key="embedding",
            embedding_dimensionality=actual_dimensions,  # Use the correct dimensions
            filterable_metadata_field_keys=safe_filterable_fields,
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents=[], storage_context=storage_context)

        return VisionQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            multi_modal_llm=azure_openai_mm_llm,  # May be None if initialization failed
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

query_engine = initialize_engine()

# ================== Utility Functions ==================
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in FrontendConfig.ALLOWED_EXTENSIONS

def get_file_type(filename):
    """Get file type from filename."""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return 'pdf'
    elif ext in {'xlsx', 'xls', 'xlsm'}:
        return 'excel'
    return 'unknown'

def validate_image_url(url: str) -> bool:
    """Verify image URL is accessible"""
    try:
        response = requests.head(url, timeout=3)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Image validation failed for {url}: {str(e)}")
        return False

# ================== Compatible Node Addition ==================
async def add_nodes_to_index_compatible(nodes: List[TextNode]):
    """Add nodes to index using only guaranteed fields and correct dimensions."""
    try:
        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )
        
        embed_model = Settings.embed_model
        available_fields = get_available_fields()
        index_dimensions = get_index_embedding_dimensions()
        documents = []
        
        logger.info(f"Using embedding dimensions: {index_dimensions}")
        
        for node in nodes:
            content = node.get_content()
            token_count = count_tokens(content)
            
            logger.info(f"Processing node {node.node_id}: {token_count} tokens")
            
            # If content is too long, chunk it
            if token_count > 7000:
                logger.info(f"Chunking large content ({token_count} tokens)")
                chunks = chunk_text(content, max_tokens=6000)
                
                for i, chunk in enumerate(chunks):
                    try:
                        embedding = embed_model.get_text_embedding(chunk)
                        
                        # Verify embedding dimensions
                        if len(embedding) != index_dimensions:
                            logger.error(f"Embedding dimension mismatch: got {len(embedding)}, expected {index_dimensions}")
                            continue
                        
                        chunk_node_id = f"{node.node_id}_chunk_{i}"
                        
                        # Enhanced metadata that includes all Excel info in JSON
                        enhanced_metadata = {
                            **node.metadata,
                            "original_node_id": node.node_id,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "is_chunked": True
                        }
                        
                        # Create document with only guaranteed fields
                        doc = {
                            "id": chunk_node_id,
                            "full_text": chunk,
                            "metadata": json.dumps(enhanced_metadata),
                            "doc_id": node.metadata.get("doc_id", ""),
                            "embedding": embedding,
                        }
                        
                        # Add optional fields only if they exist in the index
                        if "page_num" in available_fields and "page_num" in node.metadata:
                            doc["page_num"] = node.metadata["page_num"]
                        
                        if "image_path" in available_fields and "image_path" in node.metadata:
                            doc["image_path"] = node.metadata["image_path"]
                        
                        documents.append(doc)
                        logger.info(f"Created chunk {i+1}/{len(chunks)} for node {node.node_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process chunk {i} of node {node.node_id}: {e}")
                        continue
            else:
                # Content is small enough, process normally
                try:
                    embedding = embed_model.get_text_embedding(content)
                    
                    # Verify embedding dimensions
                    if len(embedding) != index_dimensions:
                        logger.error(f"Embedding dimension mismatch: got {len(embedding)}, expected {index_dimensions}")
                        continue
                    
                    # Create document with only guaranteed fields
                    doc = {
                        "id": node.node_id,
                        "full_text": content,
                        "metadata": json.dumps(node.metadata),
                        "doc_id": node.metadata.get("doc_id", ""),
                        "embedding": embedding,
                    }
                    
                    # Add optional fields only if they exist in the index
                    if "page_num" in available_fields and "page_num" in node.metadata:
                        doc["page_num"] = node.metadata["page_num"]
                    
                    if "image_path" in available_fields and "image_path" in node.metadata:
                        doc["image_path"] = node.metadata["image_path"]
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Failed to process node {node.node_id}: {e}")
                    continue
        
        # Upload documents to the search index
        if documents:
            logger.info(f"Uploading {len(documents)} documents to search index")
            result = search_client.upload_documents(documents)
            
            success_count = 0
            for item in result:
                if item.succeeded:
                    success_count += 1
                else:
                    logger.error(f"Failed to upload document {item.key}: {item.error_message}")
            
            logger.info(f"Successfully uploaded {success_count}/{len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error adding nodes to index: {str(e)}")
        raise

# ================== File Processing Functions ==================
def process_excel_file_sync(file_path: str) -> dict:
    """Process Excel file synchronously."""
    async def _process():
        excel_processor = ExcelProcessor(
            blob_connection_string=FrontendConfig.BLOB_CONNECTION_STRING,
            container_name=FrontendConfig.BLOB_CONTAINER
        )
        
        # Process Excel file
        excel_data = await excel_processor.process_excel_file(file_path)
        
        # Upload images if any
        image_urls = {}
        if excel_data.get("images"):
            image_urls = await excel_processor.upload_images_to_blob(excel_data["images"])
        
        # Create search nodes
        nodes = excel_processor.create_search_nodes(excel_data, image_urls)
        
        # Add nodes to existing index using compatible approach
        await add_nodes_to_index_compatible(nodes)
        
        return {
            'summary': excel_data.get('metadata', {}).get('title', 'Excel file processed'),
            'node_count': len(nodes),
            'sheet_count': len(excel_data.get('sheets', [])),
            'image_count': len(image_urls)
        }
    
    return asyncio.run(_process())

def process_pdf_file_sync(file_path: str) -> dict:
    """Process PDF file synchronously (placeholder for now)."""
    return {
        'summary': f"PDF processing not fully implemented yet",
        'node_count': 0,
        'page_count': 0,
        'image_count': 0
    }

# ================== API Endpoints ==================
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Process chat messages and return formatted response"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        content_filter = data.get('content_filter', 'all')
        
        if not query:
            return jsonify({'error': 'Empty query received'}), 400
        
        logger.info(f"Processing query: {query} with filter: {content_filter}")
        
        # Execute the query
        response = query_engine.custom_query(query)
        
        # Initialize response components
        pages = []
        images = []
        sheets = []
        tables = []
        source_previews = []
        data_insights = []
        
        # Process each source node
        for node in response.source_nodes:
            # Get metadata - handle both string JSON and dict formats
            metadata = node.metadata
            if isinstance(metadata.get('metadata'), str):
                try:
                    metadata_json = json.loads(metadata.get('metadata', '{}'))
                except:
                    metadata_json = metadata
            else:
                metadata_json = metadata
            
            # Extract content type and other metadata
            content_type = metadata_json.get('content_type', '')
            node_type = metadata_json.get('node_type', '')
            doc_id = metadata_json.get('doc_id', '')
            
            # Handle Excel-specific content
            if 'excel' in content_type:
                sheet_name = metadata_json.get('sheet_name', '')
                sheet_index = metadata_json.get('sheet_index', 0)
                table_name = metadata_json.get('table_name', '')
                structure_type = metadata_json.get('structure_type', 'unknown')
                
                # Add sheet info
                if node_type == 'sheet_summary' and sheet_name:
                    sheet_info = {
                        'name': sheet_name,
                        'type': structure_type
                    }
                    if sheet_info not in sheets:
                        sheets.append(sheet_info)
                
                # Add table info
                if node_type == 'table_data' and table_name:
                    table_info = {
                        'name': table_name,
                        'sheet': sheet_name,
                        'rows': metadata_json.get('row_count', 0),
                        'cols': metadata_json.get('col_count', 0)
                    }
                    if table_info not in tables:
                        tables.append(table_info)
                    
                    # Add table structure insight
                    headers = metadata_json.get('headers', '[]')
                    if isinstance(headers, str):
                        try:
                            headers = json.loads(headers)
                        except:
                            headers = []
                    
                    if headers:
                        data_insights.append({
                            'type': 'table_structure',
                            'message': f"Found table '{table_name}' in sheet '{sheet_name}' with {len(headers)} columns",
                            'details': {
                                'columns': headers[:10],  # Show first 10 columns
                                'total_columns': len(headers)
                            }
                        })
            
            # Handle PDF content
            elif content_type == 'pdf' or 'page_num' in metadata_json:
                page_num = metadata_json.get('page_num')
                if page_num and page_num not in pages:
                    pages.append(page_num)
            
            # Handle images
            image_path = metadata_json.get('image_path', '')
            if image_path:
                image_url = FrontendConfig.build_image_url(image_path)
                if image_url and image_url not in images:
                    images.append(image_url)
            
            # Create source preview
            source_preview = {
                'id': node.node_id,
                'page': metadata_json.get('page_num'),
                'content': node.get_content(metadata_mode=MetadataMode.LLM)[:250] + "...",
                'imageUrl': FrontendConfig.build_image_url(image_path) if image_path else None,
                'contentType': content_type,
                'sheetName': metadata_json.get('sheet_name'),
                'tableName': metadata_json.get('table_name'),
                'category': 'Excel' if 'excel' in content_type else 'PDF' if content_type == 'pdf' else 'Document',
                'title': metadata_json.get('file_name', doc_id),
                'date': datetime.now()
            }
            source_previews.append(source_preview)
        
        # Add summary insights
        if tables:
            total_rows = sum(t.get('rows', 0) for t in tables)
            data_insights.insert(0, {
                'type': 'summary',
                'message': f'Analyzing {len(tables)} tables with {total_rows:,} total rows across {len(sheets)} sheets'
            })
        
        # Check for relevant columns based on query
        if any(keyword in query.lower() for keyword in ['column', 'header', 'field', 'attribute']):
            relevant_columns = []
            for node in response.source_nodes:
                metadata = node.metadata
                if isinstance(metadata.get('metadata'), str):
                    try:
                        metadata_json = json.loads(metadata.get('metadata', '{}'))
                    except:
                        metadata_json = metadata
                else:
                    metadata_json = metadata
                
                headers = metadata_json.get('headers', '[]')
                if isinstance(headers, str):
                    try:
                        headers = json.loads(headers)
                    except:
                        headers = []
                
                for header in headers:
                    if any(term in header.lower() for term in query.lower().split()):
                        relevant_columns.append(header)
            
            if relevant_columns:
                data_insights.append({
                    'type': 'relevant_columns',
                    'message': f'Found {len(set(relevant_columns))} columns matching your query',
                    'details': {
                        'columns': list(set(relevant_columns))[:20]  # Show up to 20 unique columns
                    }
                })
        
        # Build the response
        response_data = {
            'response': response.response,
            'sources': {
                'pages': sorted(list(set(pages))) if pages else [],
                'images': images,
                'sheets': sheets,
                'tables': tables
            },
            'sourcePreviews': source_previews,
            'dataInsights': data_insights
        }
        
        logger.info(f"Sending response with {len(sheets)} sheets, {len(tables)} tables, {len(data_insights)} insights")
        
        return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'message': 'Failed to process your request. Please try again.'
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: PDF, Excel (.xlsx, .xls, .xlsm)'}), 400
        
        filename = secure_filename(file.filename)
        file_type = get_file_type(filename)
        
        upload_dir = Path(FrontendConfig.UPLOAD_FOLDER)
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / filename
        file.save(str(file_path))
        
        if file_type == 'excel':
            result = process_excel_file_sync(str(file_path))
        elif file_type == 'pdf':
            result = process_pdf_file_sync(str(file_path))
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        os.remove(str(file_path))
        
        return jsonify({
            'message': 'File processed successfully',
            'filename': filename,
            'file_type': file_type,
            'summary': result.get('summary', ''),
            'node_count': result.get('node_count', 0),
            'sheet_count': result.get('sheet_count'),
            'image_count': result.get('image_count')
        })
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({
            'error': 'File processing failed',
            'message': str(e)
        }), 500

@app.route('/api/chat_enhanced', methods=['POST'])
def handle_chat_enhanced():
    """Enhanced chat endpoint - uses the same logic as basic chat."""
    return handle_chat()

@app.route('/api/files', methods=['GET'])
def list_uploaded_files():
    """List all processed files and their metadata."""
    try:
        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )
        
        # Use only guaranteed fields
        results = search_client.search(
            search_text="*",
            select=["doc_id", "metadata"],
            top=100
        )
        
        files = []
        processed_doc_ids = set()
        
        for result in results:
            doc_id = result.get('doc_id', '')
            if doc_id and doc_id not in processed_doc_ids:
                processed_doc_ids.add(doc_id)
                
                # Parse metadata to extract file info
                try:
                    metadata = json.loads(result.get('metadata', '{}'))
                    if metadata.get('node_type') == 'file_summary':
                        files.append({
                            'docId': doc_id,
                            'fileName': metadata.get('file_name', doc_id),
                            'contentType': metadata.get('content_type', 'unknown'),
                            'sheetCount': metadata.get('sheet_count', 0) if metadata.get('content_type') == 'excel_file' else None
                        })
                except:
                    # Fallback for non-JSON metadata
                    files.append({
                        'docId': doc_id,
                        'fileName': doc_id,
                        'contentType': 'unknown',
                        'sheetCount': None
                    })
        
        return jsonify({'files': files})
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return jsonify({'error': 'Failed to list files'}), 500

@app.route('/api/files/<doc_id>', methods=['DELETE'])
def delete_file(doc_id: str):
    """Delete a file and all its associated nodes from the index."""
    try:
        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )
        
        results = search_client.search(
            search_text="*",
            filter=f"doc_id eq '{doc_id}'",
            select=["id"],
            top=1000
        )
        
        node_ids = [result['id'] for result in results]
        if node_ids:
            search_client.delete_documents([{"id": node_id} for node_id in node_ids])
        
        return jsonify({
            'message': f'Deleted {len(node_ids)} nodes for document {doc_id}',
            'deletedNodes': len(node_ids)
        })
        
    except Exception as e:
        logger.error(f"Error deleting file {doc_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete file'}), 500

# ================== Health Check ==================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify configuration"""
    return jsonify({
        'status': 'healthy',
        'config': {
            'azure_endpoint': FrontendConfig.AZURE_OPENAI_ENDPOINT,
            'chat_deployment': FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            'embedding_deployment': FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            'search_index': FrontendConfig.INDEX_NAME,
            'multimodal_available': azure_openai_mm_llm is not None
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)