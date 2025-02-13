import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from typing import Optional, List
import logging
import requests

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
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
    INDEX_NAME = "azure-multimodal-search"  # Matches ingest.py
    BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "rag-demo-images")
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")

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
        'AZURE_STORAGE_SAS_TOKEN'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

validate_env()

# ================== Initialize Flask ==================
app = Flask(__name__)
CORS(app)

# ================== Multimodal LLM ==================
azure_openai_mm_llm = AzureOpenAIMultiModal(
    engine=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
    api_version="2024-08-01-preview",
    model=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
    max_new_tokens=4096,
    api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
    api_base=FrontendConfig.AZURE_OPENAI_ENDPOINT,
)

# ================== Enhanced Prompt Template ==================
QA_PROMPT_TMPL = """\
You are a helpful AI assistant with access to both text and images. 
Use the document text and any associated images to provide the best possible answer.

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
    """Updated query engine matching backend improvements"""
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: AzureOpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs):
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
        
        try:
            formatted_prompt = self.qa_prompt.format(
                context_str=context_str,
                query_str=query_str
            )
            # Pass both the prompt and the images to the multi-modal LLM
            response = self.multi_modal_llm.complete(
                prompt=formatted_prompt,
                image_documents=[n.node for n in image_nodes],
            )
            
            if not response or not str(response).strip():
                raise ValueError("Empty response from OpenAI")

            # Build a references list from the nodes
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
                    "images": image_nodes
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

# ================== Initialize Query Engine ==================
def initialize_engine():
    """Initialize Azure components and query engine"""
    try:
        llm = AzureOpenAI(
            model=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-08-01-preview",
            streaming=True
        )

        embed_model = AzureOpenAIEmbedding(
            model=FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            deployment_name=FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-08-01-preview",
        )

        # Tie these to the global Settings (matching ingest.py)
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Here we use a SearchClient to READ from the existing index
        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )

        # Vector store reading from the existing index
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=SearchClient(
                endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
                index_name=FrontendConfig.INDEX_NAME,  # index name is already here
                credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
            ),
            id_field_key="id",
            chunk_field_key="full_text",
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            embedding_field_key="embedding",
            embedding_dimensionality=1536,
            filterable_metadata_field_keys={
                "page_num": ("page_num", MetadataIndexFieldType.INT64),
                "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
                "image_path": ("image_path", MetadataIndexFieldType.STRING),
                "full_text": ("full_text", MetadataIndexFieldType.STRING),
            },
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        # Load existing index (which ingest.py already populated)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents=[], storage_context=storage_context)

        return VisionQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            multi_modal_llm=azure_openai_mm_llm,
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

query_engine = initialize_engine()

# ================== Validate URL ==================
def validate_image_url(url: str) -> bool:
    """Verify image URL is accessible"""
    try:
        response = requests.head(url, timeout=3)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Image validation failed for {url}: {str(e)}")
        return False

# ================== API Endpoints ==================
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Process chat messages and return formatted response"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query received'}), 400
        
        response = query_engine.custom_query(query)
        
        # Extract and validate response components
        pages = list(response.metadata.get('pages', []))
        
        valid_images = []
        for img in response.metadata.get('images', []):
            img_url = img.node.image_url
            if img_url and validate_image_url(img_url):
                valid_images.append(img_url)
            else:
                logger.warning(f"Invalid image URL removed: {img_url}")

        # Build source previews with validation
        source_previews = []
        for node in response.source_nodes:
            image_path = node.metadata.get('image_path')
            image_url = FrontendConfig.build_image_url(image_path) if image_path else None

            if image_url and not validate_image_url(image_url):
                image_url = None

            source_previews.append({
                'page': node.metadata.get('page_num', 'N/A'),
                'content': node.get_content(metadata_mode=MetadataMode.LLM)[:250] + "...",
                'imageUrl': image_url
            })
        
        logger.debug(f"Constructed image URLs: {valid_images}")
        logger.debug(f"Source previews: {source_previews}")
            
        return jsonify({
            'response': response.response,
            'sources': {
                'pages': pages,
                'images': valid_images
            },
            'sourcePreviews': source_previews
        })
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to process your request. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
