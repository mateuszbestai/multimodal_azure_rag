import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from typing import Optional, List
import logging
import requests
import json
from urllib.parse import urlparse, unquote, quote

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType
)

# ================== Load Environment ==================
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ================== Initialize Flask ==================
app = Flask(__name__)

# ================== Configure CORS ==================
cors_config = {
    "origins": ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
    "expose_headers": ["Content-Type", "X-Accel-Buffering"],
    "supports_credentials": True,
    "send_wildcard": False,
    "vary_header": True
}

# Apply CORS only once
CORS(app, resources={
    r"/api/*": cors_config
})

# Additional headers for SSE endpoints
@app.after_request
def after_request(response):
    if request.path == '/api/chat/stream':
        response.headers['Cache-Control'] = 'no-cache, no-transform'
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Connection'] = 'keep-alive'
    return response

# ================== Configuration Class ==================
class FrontendConfig:
    """Centralized configuration for frontend components"""
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    SEARCH_SERVICE_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    INDEX_NAME = "azure-multimodal-search-new"
    BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "rag-test")  # Match ingest.py
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")

    @classmethod
    def build_image_url(cls, blob_path: str) -> str:
        """Build Azure Blob Storage URL with SAS token"""
        if not blob_path:
            return ""
        
        # If already a full URL, return it (optionally with SAS if needed)
        if blob_path.startswith("http"):
            # It's already a full URL - check if it needs SAS token
            if '?' in blob_path:
                # Already has query params (probably SAS), return as-is
                return blob_path
            
            # Check if it's our blob storage and we have a SAS token
            if f"{cls.STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{cls.BLOB_CONTAINER}" in blob_path:
                # It's our blob URL without SAS
                if cls.SAS_TOKEN:
                    # Add SAS token
                    sas = cls.SAS_TOKEN if cls.SAS_TOKEN.startswith('?') else f'?{cls.SAS_TOKEN}'
                    return f"{blob_path}{sas}"
            
            # Return the URL as-is (it's accessible with anonymous access)
            return blob_path
        
        # It's just a blob name (like "page_1.png"), build the full URL
        clean_path = blob_path.lstrip('/')
        
        # Don't encode if it looks like it's already encoded
        if '%' not in clean_path:
            from urllib.parse import quote
            encoded_path = quote(clean_path, safe='/')
        else:
            encoded_path = clean_path
        
        # Build the base URL
        base_url = f"https://{cls.STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{cls.BLOB_CONTAINER}/{encoded_path}"
        
        # Add SAS token if available
        if cls.SAS_TOKEN:
            sas = cls.SAS_TOKEN if cls.SAS_TOKEN.startswith('?') else f'?{cls.SAS_TOKEN}'
            return f"{base_url}{sas}"
        else:
            # If no SAS token but container is public, URL should still work
            return base_url

def validate_env():
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_SEARCH_SERVICE_ENDPOINT',
        'AZURE_SEARCH_ADMIN_KEY',
        'AZURE_STORAGE_ACCOUNT_NAME',
        'AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME',
        'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    
    logger.info(f"Azure OpenAI Endpoint: {FrontendConfig.AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Chat Deployment Name: {FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT}")
    logger.info(f"Embedding Deployment Name: {FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")

validate_env()

# ================== Multimodal LLM ==================
# Initialize multimodal LLM with correct parameters
try:
    azure_openai_mm_llm = AzureOpenAIMultiModal(
        deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version="2024-02-01",  # Use stable API version
        api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
        azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
        max_new_tokens=4096,
    )
    logger.info("Multimodal LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize multimodal LLM: {str(e)}")
    raise

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
    """Query engine for multimodal RAG"""
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: AzureOpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs):
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str) -> LlamaResponse:
        try:
            nodes = self.retriever.retrieve(query_str)
            
            # Build image nodes with page number references
            image_nodes = []
            for n in nodes:
                blob_path = n.metadata.get("image_path")
                if blob_path:
                    try:
                        full_url = FrontendConfig.build_image_url(blob_path)
                        if full_url:
                            # Validate the URL is accessible before adding
                            logger.debug(f"Testing image URL: {full_url}")
                            img_node = ImageNode(image_url=full_url)
                            img_node.metadata = {"page_num": n.metadata.get("page_num", "N/A")}
                            image_nodes.append(NodeWithScore(node=img_node))
                    except Exception as e:
                        logger.error(f"Image node error for {blob_path}: {str(e)}")
            
            # Build the textual context
            context_str = "\n".join([
                f"Page {n.metadata.get('page_num', '?')}: {n.get_content(metadata_mode=MetadataMode.LLM)}"
                for n in nodes
            ])
            
            formatted_prompt = self.qa_prompt.format(
                context_str=context_str,
                query_str=query_str
            )
            
            logger.info(f"Attempting query with {len(image_nodes)} images")
            
            # Try multimodal completion first if we have images
            response = None
            if image_nodes:
                try:
                    logger.debug("Attempting multimodal completion...")
                    response = self.multi_modal_llm.complete(
                        prompt=formatted_prompt,
                        image_documents=[n.node for n in image_nodes],
                    )
                except Exception as mm_error:
                    logger.warning(f"Multimodal completion failed: {str(mm_error)}")
                    # Check if it's an access error
                    if "403" in str(mm_error) or "can not be accessed" in str(mm_error):
                        logger.warning("Image access denied - falling back to text-only mode")
                    else:
                        logger.error(f"Unexpected multimodal error: {str(mm_error)}")
            
            # Fallback to text-only if multimodal failed or no images
            if not response:
                logger.info("Using text-only completion...")
                response = Settings.llm.complete(formatted_prompt)
            
            if not response or not str(response).strip():
                raise ValueError("Empty response from OpenAI")

            # Build references
            references = []
            for n in nodes:
                ref_text = f"Page {n.metadata.get('page_num', 'N/A')}: {n.get_content(metadata_mode=MetadataMode.LLM)[:100]}..."
                if n.metadata.get("image_path"):
                    ref_text += " [Image available]"
                references.append(ref_text)
            
            return LlamaResponse(
                response=str(response),
                source_nodes=nodes,
                metadata={
                    "references": references,
                    "pages": list({int(n.metadata.get("page_num", 0)) for n in nodes if n.metadata.get("page_num")}),
                    "images": [img.node.image_url for img in image_nodes]
                }
            )
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            raise

    def stream_query(self, query_str: str):
        """Stream the response for real-time display"""
        try:
            nodes = self.retriever.retrieve(query_str)
            
            # Build image nodes
            image_nodes = []
            for n in nodes:
                blob_path = n.metadata.get("image_path")
                if blob_path:
                    try:
                        full_url = FrontendConfig.build_image_url(blob_path)
                        if full_url:
                            logger.debug(f"Testing image URL: {full_url}")
                            img_node = ImageNode(image_url=full_url)
                            img_node.metadata = {"page_num": n.metadata.get("page_num", "N/A")}
                            image_nodes.append(NodeWithScore(node=img_node))
                    except Exception as e:
                        logger.error(f"Image node error for {blob_path}: {str(e)}")
            
            # Build context
            context_str = "\n".join([
                f"Page {n.metadata.get('page_num', '?')}: {n.get_content(metadata_mode=MetadataMode.LLM)}"
                for n in nodes
            ])
            
            formatted_prompt = self.qa_prompt.format(
                context_str=context_str,
                query_str=query_str
            )
            
            logger.info(f"Attempting streaming with {len(image_nodes)} images")
            
            # Try streaming with multimodal if we have images
            response_gen = None
            if image_nodes:
                try:
                    logger.debug("Attempting multimodal streaming...")
                    response_gen = self.multi_modal_llm.stream_complete(
                        prompt=formatted_prompt,
                        image_documents=[n.node for n in image_nodes],
                    )
                except Exception as mm_error:
                    logger.warning(f"Multimodal streaming failed: {str(mm_error)}")
                    if "403" in str(mm_error) or "can not be accessed" in str(mm_error):
                        logger.warning("Image access denied - falling back to text-only streaming")
                    else:
                        logger.error(f"Unexpected multimodal error: {str(mm_error)}")
            
            # Fallback to text-only streaming
            if not response_gen:
                logger.info("Using text-only streaming...")
                response_gen = Settings.llm.stream_complete(formatted_prompt)
            
            # Extract metadata
            pages = list({int(n.metadata.get("page_num", 0)) for n in nodes if n.metadata.get("page_num")})
            valid_images = [img.node.image_url for img in image_nodes]

            # Build source previews
            source_previews = []
            for node in nodes:
                image_path = node.metadata.get('image_path')
                image_url = None
                if image_path:
                    try:
                        image_url = FrontendConfig.build_image_url(image_path)
                    except Exception as e:
                        logger.warning(f"Failed to build image URL for preview: {e}")

                source_previews.append({
                    'page': node.metadata.get('page_num', 'N/A'),
                    'content': node.get_content(metadata_mode=MetadataMode.LLM)[:250] + "...",
                    'imageUrl': image_url
                })
            
            return response_gen, {
                'pages': pages,
                'images': valid_images,
                'sourcePreviews': source_previews
            }
        except Exception as e:
            logger.error(f"Stream query error: {str(e)}", exc_info=True)
            raise

# ================== Initialize Query Engine ==================
def initialize_engine():
    """Initialize Azure components and query engine"""
    try:
        # Initialize LLM
        llm = AzureOpenAI(
            deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",
            streaming=True
        )

        # Initialize embedding model
        embed_model = AzureOpenAIEmbedding(
            deployment_name=FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",
        )

        # Test the deployments
        logger.info("Testing LLM deployment...")
        test_response = llm.complete("Hello")
        logger.info(f"LLM test successful: {test_response}")

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Initialize vector store
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=SearchClient(
                endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
                index_name=FrontendConfig.INDEX_NAME,
                credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
            ),
            id_field_key="id",
            chunk_field_key="full_text",
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            embedding_field_key="embedding",
            embedding_dimensionality=3072,  # Updated for text-embedding-3-large
            filterable_metadata_field_keys={
                "page_num": ("page_num", MetadataIndexFieldType.INT64),
                "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
                "image_path": ("image_path", MetadataIndexFieldType.STRING),
                "full_text": ("full_text", MetadataIndexFieldType.STRING),
            },
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        
        # Load existing index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents=[], storage_context=storage_context)

        return VisionQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            multi_modal_llm=azure_openai_mm_llm,
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        raise

# Initialize query engine
try:
    query_engine = initialize_engine()
    logger.info("Query engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize query engine: {str(e)}")
    raise

# ================== API Endpoints ==================
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Process chat messages and return formatted response"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query received'}), 400
        
        logger.info(f"Processing query: {query}")
        
        response = query_engine.custom_query(query)
        
        # Extract and validate response components
        pages = list(response.metadata.get('pages', []))
        valid_images = response.metadata.get('images', [])
        
        # Build source previews with proper image URLs
        source_previews = []
        for node in response.source_nodes:
            image_path = node.metadata.get('image_path')
            image_url = FrontendConfig.build_image_url(image_path) if image_path else None
            
            # Log for debugging
            if image_url:
                logger.debug(f"Source preview image URL: {image_url}")

            source_previews.append({
                'page': node.metadata.get('page_num', 'N/A'),
                'content': node.get_content(metadata_mode=MetadataMode.LLM)[:250] + "...",
                'imageUrl': image_url  # Make sure this is included
            })
        
        logger.info(f"Query processed successfully. Pages: {pages}, Images: {len(valid_images)}, Source previews: {len(source_previews)}")
        
        # Log the first source preview for debugging
        if source_previews:
            logger.debug(f"First source preview: {source_previews[0]}")
            
        return jsonify({
            'response': response.response,
            'sources': {
                'pages': pages,
                'images': valid_images
            },
            'sourcePreviews': source_previews
        })
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'message': 'Failed to process your request. Please check the logs for details.'
        }), 500

@app.route('/api/chat/stream', methods=['GET'])
def handle_chat_stream():
    """Stream chat responses using Server-Sent Events"""
    query = request.args.get('message', '').strip()
    
    if not query:
        return jsonify({'error': 'Empty query received'}), 400
    
    def generate():
        try:
            logger.info(f"Streaming query: {query}")
            
            # Get the streaming response
            response_gen, metadata = query_engine.stream_query(query)
            
            # Log metadata for debugging
            logger.debug(f"Stream metadata: {metadata}")
            if metadata.get('sourcePreviews'):
                logger.debug(f"First source preview in stream: {metadata['sourcePreviews'][0]}")
            
            # Send metadata first
            yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
            
            # Stream the response chunks
            for chunk in response_gen:
                if chunk.delta:
                    yield f"data: {json.dumps({'type': 'chunk', 'data': chunk.delta})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify configuration"""
    try:
        # Test Azure Search connection
        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )
        doc_count = search_client.get_document_count()
        
        return jsonify({
            'status': 'healthy',
            'config': {
                'azure_endpoint': FrontendConfig.AZURE_OPENAI_ENDPOINT,
                'chat_deployment': FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
                'embedding_deployment': FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                'search_index': FrontendConfig.INDEX_NAME,
                'document_count': doc_count
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)