# --- Azure Authentication ---
# --- Q1 ---

# When we run a Python script locally that uses DefaultAzureCredential, it usually relies on our Azure CLI login credentials for authentication.
# Before running the script, we must run "az login" to sign in to Azure.
# DefaultAzureCredential automatically checks for available credentials, including Azure CLI credentials, and uses them if they are available.

# --- Q2 --- 
# A deployed pipeline running on an Azure VM or container cannot use "az login" because there is no user available to interactively sign in.
# Instead, it typically uses a Managed Identity assigned to the resource.
# DefaultAzureCredential can automatically detect and use the Managed Identity, which allows the same Python code to work locally and in Azure  without requiring code changes.

# --- Q3 ----
# If a script immediately gets an AuthenticationError, one likely cause is that we have not run "az login" locally, so no Azure credentials are available.
# We can diagnose this by running "az account show" to verify that we are logged in.
# Another likely cause is that the account or Managed Identity does not have permission to access the Azure resource being requested. 
# We can diagnose this by checking the error message and verifying the assigned roles and permissions in Azure.

# --- Blob Storage ---
# --- Q1 ---
# Azure Blob Storage has three levels: storage account, container, and blob.
# A storage account is like a filing cabinet, a container is like a drawer inside the cabinet, and a blob is like an individual file stored in that drawer.

# --- Q2 ---
# Scenario 1: A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
# I would use Blob Storage because I need to save raw JSON responses as files that can be reprocessed later.

# Scenario 2: Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
# I would use a relational database because the analytics team needs to query large amounts of structured data by date range and customer ID.

# Scenario 3: A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.
# I would use Blob Storage because NumPy arrays are file-like objects that can be stored and retrieved between pipeline runs.

# --- Q3 ---
def list_container(container_client):
    """
    Print the name and size of every blob in the container.
    """
    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")


# --- Q4 ---
def upload_text(container_client, blob_name, text):
    """
    Upload a UTF-8 encoded text string as a blob, overwriting if it exists.
    """
    data = text.encode("utf-8")
    container_client.upload_blob(
        name=blob_name,
        data=data,
        overwrite=True
    )