import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI, APIConnectionError, AuthenticationError, NotFoundError, RateLimitError, APIStatusError

load_dotenv()

def check_azure_deployments():
    """Check your Azure OpenAI deployments and test connectivity."""
    
    print("🔍 Checking Azure OpenAI Configuration...")
    print("=" * 60)
    
    # Get environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    # User might set a specific API version in their .env
    api_version_env = os.getenv("AZURE_OPENAI_API_VERSION") 
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    
    print(f"📍 AZURE_OPENAI_ENDPOINT: {endpoint}")
    print(f"🔑 AZURE_OPENAI_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
    print(f" SCRIPT DEFAULT API VERSION (for SDK tests): 2024-08-01-preview") # Default for this script
    print(f"🔧 AZURE_OPENAI_API_VERSION (from .env): {api_version_env if api_version_env else 'Not Set (will use script default or try fallbacks)'}")
    print(f"🤖 AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME: {chat_deployment}")
    print(f"🔤 AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME: {embedding_deployment if embedding_deployment else 'Not Set'}")
    print()
    
    if not all([endpoint, api_key, chat_deployment]):
        print("❌ CRITICAL: Missing one or more required environment variables (ENDPOINT, API_KEY, CHAT_DEPLOYMENT_NAME)!")
        print("   Please ensure these are correctly set in your .env file or environment.")
        return False
    
    # Test 1: Check if endpoint is reachable and list deployments
    print("🌐 [Test 1] Testing endpoint connectivity and listing deployments...")
    print("-" * 50)
    base_url = endpoint.rstrip('/')
    # Using a generally stable API version for listing deployments
    deployments_list_api_version = "2023-05-15" 
    test_url = f"{base_url}/openai/deployments?api-version={deployments_list_api_version}"
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    all_deployments_found = []
    try:
        response = requests.get(test_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            print(f"✅ Endpoint is reachable (HTTP {response.status_code}) using api-version '{deployments_list_api_version}'.")
            deployments_data = response.json()
            all_deployments_found = deployments_data.get('data', [])
            
            print(f"\n📋 Found {len(all_deployments_found)} deployments:")
            for deployment in all_deployments_found:
                deployment_id = deployment.get('id', 'Unknown ID')
                model_name = deployment.get('model', 'Unknown model')
                status = deployment.get('status', 'Unknown status')
                print(f"   - Name: '{deployment_id}', Model: '{model_name}', Status: '{status}'")
            
            deployment_names = [d.get('id', '') for d in all_deployments_found]
            if chat_deployment in deployment_names:
                print(f"✅ Your chat deployment '{chat_deployment}' IS LISTED by the Azure resource.")
            else:
                print(f"❌ Your chat deployment '{chat_deployment}' IS NOT LISTED by the Azure resource.")
                print(f"   Available deployments found: {deployment_names}")
                print("   Double-check the name in your .env file against the names listed above (case-sensitive).")
                # return False # Let's continue to SDK tests, as sometimes listing fails but SDK works
                
        else:
            print(f"❌ Endpoint test failed with requests: HTTP {response.status_code}")
            print(f"   URL: {test_url}")
            print(f"   Response: {response.text[:500]}...") # Print first 500 chars of error
            print(f"   This could be due to: incorrect API key, endpoint URL, network issues, or RBAC permissions on the Azure OpenAI resource.")
            return False # Critical failure
            
    except requests.exceptions.Timeout:
        print(f"❌ Endpoint test failed: Request timed out after 15 seconds.")
        print(f"   URL: {test_url}")
        print(f"   Check your network connection and if the endpoint '{endpoint}' is correct and accessible.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Endpoint test failed with requests: {type(e).__name__} - {str(e)}")
        print(f"   URL: {test_url}")
        print(f"   This often indicates a problem with the endpoint URL format or network connectivity.")
        return False
    
    # Test 2: Test chat completion
    print(f"\n💬 [Test 2] Testing chat completion with deployment '{chat_deployment}'...")
    print("-" * 50)

    # Prioritize API version from .env, then script default, then fallbacks
    initial_api_version_to_try = api_version_env if api_version_env else "2024-08-01-preview"
    api_versions_to_test = [initial_api_version_to_try] + [
        v for v in ["2024-06-01", "2024-05-01-preview", "2024-02-15-preview", "2024-02-01", "2023-12-01-preview", "2023-07-01-preview"] 
        if v != initial_api_version_to_try # Avoid duplicates
    ]
    
    chat_test_succeeded = False
    successful_chat_api_version = None

    for current_api_version in api_versions_to_test:
        print(f"   Attempting with API version: '{current_api_version}'...")
        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=current_api_version,
                azure_endpoint=endpoint
            )
            
            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[{"role": "user", "content": "Hello! This is a test. Reply with 'Test successful!'"}],
                max_tokens=50,
                temperature=0.1
            )
            
            print(f"   ✅ Chat completion test SUCCESSFUL with API version '{current_api_version}'!")
            print(f"   Response: {response.choices[0].message.content}")
            chat_test_succeeded = True
            successful_chat_api_version = current_api_version
            break # Success, no need to try other API versions
            
        except AuthenticationError as e:
            print(f"   ❌ API version '{current_api_version}' FAILED: AuthenticationError")
            print(f"      Message: {e.message}")
            print(f"      This means your API Key is invalid, expired, or does not have permissions for this operation/deployment.")
            # No point trying other API versions if auth fails
            return False 
        except NotFoundError as e:
            print(f"   ❌ API version '{current_api_version}' FAILED: NotFoundError")
            print(f"      Message: {e.message}")
            print(f"      This often means the deployment name '{chat_deployment}' is incorrect for this API version, or the model doesn't exist/isn't deployed.")
            print(f"      It can also happen if the API version doesn't support the 'chat completions' operation for your model type.")
            # Continue to try other API versions, as some versions might not find a specific deployment.
        except APIConnectionError as e:
            print(f"   ❌ API version '{current_api_version}' FAILED: APIConnectionError")
            print(f"      Message: {e.message}")
            print(f"      Could be a network issue, incorrect endpoint, or firewall blocking the connection to Azure OpenAI.")
            return False # Usually a fundamental connection problem
        except RateLimitError as e:
            print(f"   ❌ API version '{current_api_version}' FAILED: RateLimitError")
            print(f"      Message: {e.message}")
            print(f"      You've exceeded your quota. Check your usage and limits in the Azure portal.")
            # May or may not be recoverable by changing API version, but good to stop.
            return False
        except APIStatusError as e: # Catches other API errors (like 400, 429, 500s from Azure)
            print(f"   ❌ API version '{current_api_version}' FAILED: APIStatusError (HTTP {e.status_code})")
            print(f"      Message: {e.message}")
            if e.response and e.response.text:
                 print(f"      Azure Response: {e.response.text[:300]}...")
            # Some API versions might be more lenient or handle requests differently.
        except Exception as e:
            print(f"   ❌ API version '{current_api_version}' FAILED with an unexpected error: {type(e).__name__}")
            print(f"      Details: {str(e)}")
            # Continue to try other API versions
    
    if not chat_test_succeeded:
        print(f"\n   ❌ Chat completion test FAILED for all attempted API versions.")
        # return False # Let's continue to embedding test if configured

    # Test 3: Test embedding if deployment is set
    embedding_test_succeeded = False
    if embedding_deployment:
        print(f"\n🔤 [Test 3] Testing embedding with deployment '{embedding_deployment}'...")
        print("-" * 50)

        # Use the successful chat API version if available, or the same logic as chat
        initial_emb_api_version = successful_chat_api_version if successful_chat_api_version else (api_version_env if api_version_env else "2024-08-01-preview")
        emb_api_versions_to_test = [initial_emb_api_version] + [
            v for v in ["2024-06-01", "2024-05-01-preview", "2024-02-15-preview", "2024-02-01", "2023-12-01-preview", "2023-07-01-preview"] 
            if v != initial_emb_api_version
        ]

        for current_api_version in emb_api_versions_to_test:
            print(f"   Attempting with API version: '{current_api_version}'...")
            try:
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=current_api_version,
                    azure_endpoint=endpoint
                )
                
                response = client.embeddings.create(
                    model=embedding_deployment, # This should be the DEPLOYMENT NAME
                    input="This is a test embedding."
                )
                
                embedding_dim = len(response.data[0].embedding)
                print(f"   ✅ Embedding test SUCCESSFUL with API version '{current_api_version}'! Dimensions: {embedding_dim}")
                embedding_test_succeeded = True
                break
                
            except AuthenticationError as e:
                print(f"   ❌ API version '{current_api_version}' FAILED: AuthenticationError - {e.message}")
                return False # Critical
            except NotFoundError as e:
                print(f"   ❌ API version '{current_api_version}' FAILED: NotFoundError - {e.message}")
                print(f"      Check if embedding deployment '{embedding_deployment}' is correct and available for this API version.")
            except APIConnectionError as e:
                print(f"   ❌ API version '{current_api_version}' FAILED: APIConnectionError - {e.message}")
                return False # Critical
            except RateLimitError as e:
                print(f"   ❌ API version '{current_api_version}' FAILED: RateLimitError - {e.message}")
                return False # Critical
            except APIStatusError as e:
                print(f"   ❌ API version '{current_api_version}' FAILED: APIStatusError (HTTP {e.status_code}) - {e.message}")
                if e.response and e.response.text:
                    print(f"      Azure Response: {e.response.text[:300]}...")
            except Exception as e:
                print(f"   ❌ API version '{current_api_version}' FAILED with an unexpected error: {type(e).__name__} - {str(e)}")

        if not embedding_test_succeeded:
             print(f"\n   ❌ Embedding test FAILED for all attempted API versions.")
    else:
        print("\nℹ️ Embedding deployment name not set in .env, skipping embedding test.")

    print("\n" + "=" * 60)
    final_success = chat_test_succeeded and (embedding_test_succeeded if embedding_deployment else True)
    if final_success:
        print("✅🏁 Core diagnostic tests passed successfully!")
        if successful_chat_api_version:
            print(f"   💡 Recommended API Version for your application: '{successful_chat_api_version}'")
    else:
        print("❌🏁 One or more diagnostic tests failed.")
    
    return final_success

def fix_suggestions():
    """Provide fix suggestions based on common issues."""
    print("\n🔧 Common Azure OpenAI Troubleshooting Steps & Fixes:")
    print("-" * 50)
    print("1.  **Environment Variables (.env file):**")
    print("    *   `AZURE_OPENAI_ENDPOINT`: Should be like `https://<your-resource-name>.openai.azure.com/` (ensure trailing slash).")
    print("    *   `AZURE_OPENAI_API_KEY`: Double-check for typos or if it's for the correct resource.")
    print("    *   `AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME`: This is your *deployment name* in Azure AI Studio (e.g., 'gpt-4o-test'), NOT the model name (e.g., 'gpt-4o'). Case-sensitive!")
    print("    *   `AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME`: Same as above, for your embedding deployment (e.g., 'ada-embed-test'). Case-sensitive!")
    print("    *   `AZURE_OPENAI_API_VERSION`: If set, ensure it's a valid version (e.g., '2024-02-01'). If not set, the script tries defaults.")
    print("\n2.  **Azure Portal Checks:**")
    print("    *   **Deployment Status:** In Azure AI Studio (or Azure OpenAI Studio), go to 'Deployments'. Check: ")
    print("        - Is your deployment listed?")
    print("        - Is the 'Status' 'Succeeded'? If it's 'Updating' or 'Failed', address that first.")
    print("        - Note the exact 'Deployment name' and compare with your .env file (case-sensitive).")
    print("    *   **API Keys:** Under 'Keys and Endpoint' for your Azure OpenAI resource, verify the key you're using.")
    print("    *   **Resource Region:** Ensure your resource is in a region that supports the models you're trying to deploy.")
    print("    *   **Quotas:** Check 'Quotas' under your Azure OpenAI resource to ensure you haven't hit token or request limits.")
    print("\n3.  **API Version Mismatches:**")
    print("    *   The script attempts various API versions. If a specific one worked, consider setting `AZURE_OPENAI_API_VERSION` in your `.env` to that version.")
    print("    *   Some features/models are only available with newer API versions. Conversely, very new preview API versions can sometimes have issues.")
    print("\n4.  **Permissions (RBAC):**")
    print("    *   If using Managed Identity or Service Principal, ensure it has the 'Cognitive Services OpenAI User' role (or similar) on the Azure OpenAI resource.")
    print("    *   For API Key auth, this is less common unless key permissions were restricted somehow.")
    print("\n5.  **Network Issues:**")
    print("    *   **Firewalls/NSGs:** If running in a corporate network or Azure VNet, ensure outbound traffic to `*.openai.azure.com` on port 443 is allowed.")
    print("    *   **Private Endpoints:** If your Azure OpenAI resource uses private endpoints, ensure your client machine/environment can resolve and reach it (e.g., proper DNS, VNet peering/connectivity).")
    print("\n6.  **Specific Error Messages from the Script:**")
    print("    *   `AuthenticationError`: Almost always an API key issue or the key lacks permission for the deployment.")
    print("    *   `NotFoundError`: Often a typo in the deployment name, or the API version used doesn't recognize that deployment/operation. Or the deployment doesn't exist / failed to deploy.")
    print("    *   `APIConnectionError`: Network problem, or malformed `AZURE_OPENAI_ENDPOINT`.")
    print("    *   `RateLimitError`: You've hit usage limits. Wait or request a quota increase.")
    print("    *   `APIStatusError` (with HTTP codes like 400, 401, 403, 404, 429, 500):")
    print("        - `400 Bad Request`: Often malformed request (e.g., API version not supporting a parameter, or model not supporting the operation).")
    print("        - `401 Unauthorized`: API Key issue.")
    print("        - `403 Forbidden`: Key valid, but no permission for this specific action/deployment.")
    print("        - `404 Not Found`: Deployment name usually.")
    print("        - `429 Too Many Requests`: Rate limit or quota.")
    print("        - `500 Internal Server Error`: Problem on Azure's side; try again later.")

if __name__ == "__main__":
    # Create a dummy .env file if it doesn't exist for testing
    if not os.path.exists(".env"):
        print("ℹ️ .env file not found. Creating a .env.example file for you.")
        print("   Please populate it with your Azure OpenAI details before running again.")
        with open(".env.example", "w") as f:
            f.write("AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/\n")
            f.write("AZURE_OPENAI_API_KEY=YOUR_API_KEY\n")
            f.write("# Optional: Set a specific API version if needed, otherwise script tries defaults\n")
            f.write("# AZURE_OPENAI_API_VERSION=2024-02-01\n")
            f.write("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME=YOUR_CHAT_DEPLOYMENT_NAME\n")
            f.write("# Optional: If you use embeddings\n")
            f.write("# AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME=YOUR_EMBEDDING_DEPLOYMENT_NAME\n")
        exit()

    success = check_azure_deployments()
    if not success:
        fix_suggestions()
    else:
        print("\nIf you still face issues in your main application:")
        print("  - Ensure your application code uses the same environment variables and a successful API version found by this script.")
        print("  - Double-check that the `model` parameter in your application's OpenAI calls is set to the *deployment name*, not the base model name.")