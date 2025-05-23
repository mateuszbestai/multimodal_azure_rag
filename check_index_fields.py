import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = "azure-multimodal-search-new"

def check_index_fields():
    """Check what fields exist in the current index."""
    try:
        # Initialize search index client
        credential = AzureKeyCredential(SEARCH_SERVICE_KEY)
        index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=credential)
        
        # Get the current index
        existing_index = index_client.get_index(INDEX_NAME)
        
        print(f"📋 Current fields in index '{INDEX_NAME}':")
        print("=" * 50)
        
        existing_fields = set()
        for field in existing_index.fields:
            existing_fields.add(field.name)
            print(f"✅ {field.name}: {field.type}")
        
        print("\n" + "=" * 50)
        
        # Check for required Excel fields
        required_excel_fields = [
            'content_type', 'sheet_name', 'table_name', 'node_type',
            'sheet_index', 'table_index', 'row_count', 'col_count',
            'structure_type', 'headers', 'file_name', 'sheet_count',
            'image_filename'
        ]
        
        missing_fields = []
        for field in required_excel_fields:
            if field not in existing_fields:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing Excel fields: {', '.join(missing_fields)}")
            print("\n🔧 To fix this, run: python update_index_schema.py")
            return False
        else:
            print("✅ All required Excel fields are present!")
            return True
        
    except Exception as e:
        print(f"❌ Error checking index: {str(e)}")
        return False

if __name__ == "__main__":
    check_index_fields()