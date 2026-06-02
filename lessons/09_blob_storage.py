from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url="https://<account>.blob.core.windows.net",
    container_name="my-container",
    credential=credential
)