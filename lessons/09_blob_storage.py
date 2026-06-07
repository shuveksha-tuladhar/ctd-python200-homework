from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url="https://shuvekshactd2026sa.blob.core.windows.net",
    container_name="my-container",
    credential=credential
)

container.upload_blob("hello.txt", b"hello world", overwrite=True)
for blob in container.list_blobs():
    print(blob.name, blob.size)
    
data = container.download_blob("hello.txt").readall()
print(data)  # b'hello world'

# Encoding before upload
text = "time,temp\n2024-01-15T00:00,12.3\n2024-01-15T01:00,11.8"
container.upload_blob("data.csv", text.encode("utf-8"), overwrite=True)

# Decoding after download
raw = container.download_blob("data.csv").readall()
text = raw.decode("utf-8")
print(text)