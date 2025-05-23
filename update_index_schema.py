import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = "azure-multimodal-search-new"

def create_vector_search_config():
    """Create the vector search configuration."""
    return VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": VectorSearchAlgorithmMetric.COSINE,
                },
            )
        ],
    )

def create_semantic_search_config():
    """Create the semantic search configuration."""
    return SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="full_text")],
                    keywords_fields=[SemanticField(field_name="metadata")],
                ),
            )
        ]
    )

def add_missing_fields_to_index():
    """Add missing Excel fields to the existing index without recreating it."""
    try:
        # Initialize search index client
        credential = AzureKeyCredential(SEARCH_SERVICE_KEY)
        index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=credential)
        
        # Get the current index
        try:
            existing_index = index_client.get_index(INDEX_NAME)
            logger.info(f"Found existing index: {INDEX_NAME}")
        except Exception as e:
            logger.error(f"Index {INDEX_NAME} does not exist. Please run ingest.py first to create the base index.")
            return False
        
        # Get existing field names
        existing_field_names = {field.name for field in existing_index.fields}
        logger.info(f"Existing fields: {existing_field_names}")
        
        # Define new Excel fields to add
        new_excel_fields = [
            SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sheet_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="table_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="node_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sheet_index", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="table_index", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="row_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="col_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="structure_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="headers", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sheet_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="image_filename", type=SearchFieldDataType.String, filterable=True),
        ]
        
        # Filter out fields that already exist
        fields_to_add = [field for field in new_excel_fields if field.name not in existing_field_names]
        
        if not fields_to_add:
            logger.info("All Excel fields already exist in the index!")
            return True
        
        logger.info(f"Adding {len(fields_to_add)} new fields: {[f.name for f in fields_to_add]}")
        
        # Create the complete field list (existing + new)
        all_fields = list(existing_index.fields) + fields_to_add
        
        # Create updated index with all fields
        updated_index = SearchIndex(
            name=INDEX_NAME,
            fields=all_fields,
            semantic_search=existing_index.semantic_search,
            vector_search=existing_index.vector_search,
            cors_options=existing_index.cors_options,
            scoring_profiles=existing_index.scoring_profiles,
            default_scoring_profile=existing_index.default_scoring_profile,
            analyzers=existing_index.analyzers,
            tokenizers=existing_index.tokenizers,
            token_filters=existing_index.token_filters,
            char_filters=existing_index.char_filters,
            normalizers=existing_index.normalizers,
            encryption_key=existing_index.encryption_key
        )
        
        # Update the index
        logger.info("Updating index schema...")
        result = index_client.create_or_update_index(updated_index)
        logger.info(f"Index schema updated successfully: {result.name}")
        
        # List all fields to verify
        logger.info("Updated index fields:")
        for field in result.fields:
            logger.info(f"  - {field.name}: {field.type}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating index schema: {str(e)}")
        return False

def create_new_index_if_needed():
    """Create a completely new index if the existing one has issues."""
    try:
        # Initialize search index client
        credential = AzureKeyCredential(SEARCH_SERVICE_KEY)
        index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=credential)
        
        logger.info("Creating new index with all required fields...")
        
        # Define all fields including existing and new ones
        fields = [
            # Core fields
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="full_text", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="metadata", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            
            # Vector field with proper configuration
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile"
            ),
            
            # Existing PDF fields
            SimpleField(name="page_num", type=SearchFieldDataType.Int64, filterable=True, sortable=True, facetable=True),
            SimpleField(name="image_path", type=SearchFieldDataType.String, filterable=True),
            
            # New Excel-specific fields
            SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sheet_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="table_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="node_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sheet_index", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="table_index", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="row_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="col_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="structure_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="headers", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sheet_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
            SimpleField(name="image_filename", type=SearchFieldDataType.String, filterable=True),
        ]
        
        # Create the index
        new_index = SearchIndex(
            name=f"{INDEX_NAME}-new",
            fields=fields,
            vector_search=create_vector_search_config(),
            semantic_search=create_semantic_search_config()
        )
        
        # Create the new index
        result = index_client.create_or_update_index(new_index)
        logger.info(f"New index created successfully: {result.name}")
        
        print(f"✅ Created new index: {result.name}")
        print("📝 To use this new index:")
        print(f"   1. Update your .env file: INDEX_NAME={result.name}")
        print("   2. Re-run your ingest.py script to populate the new index")
        print("   3. Update app.py to use the new index name")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating new index: {str(e)}")
        return False

def main():
    """Main function to update or create index schema."""
    print("🔍 Checking current index status...")
    
    # First try to add missing fields to existing index
    success = add_missing_fields_to_index()
    
    if not success:
        print("\n⚠️  Could not update existing index.")
        user_input = input("\n🔧 Would you like to create a new index with all required fields? (y/n): ")
        
        if user_input.lower() in ['y', 'yes']:
            success = create_new_index_if_needed()
        else:
            print("❌ Skipping index creation.")
            return False
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Index schema updated successfully!")
        print("You can now upload Excel files.")
    else:
        print("\n❌ Failed to update index schema.")
        print("Please check the logs and try again.")