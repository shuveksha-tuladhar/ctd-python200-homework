# Part 2: Project -- Extract + Load Pipeline

# Video link:
# https://youtu.be/JYERBYpFU4k
 
from datetime import date
import json
import os

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# setup
ACCOUNT_URL = "https://shuvekshactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Step 1: Extract
url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=35.2271"
    "&longitude=-80.8431"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
)

response = requests.get(url)
response.raise_for_status()
data = response.json()

# Step 2: Serialize
json_bytes = json.dumps(data).encode("utf-8")

# Azure authentication
credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
)

container_client = blob_service_client.get_container_client(CONTAINER)

# Step 3: Load
today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

container_client.upload_blob(
    name=blob_path,
    data=json_bytes,
    overwrite=True
)

print(f"Uploaded {blob_path} ({len(json_bytes)} bytes)")

# Step 4: Verify
print("\nBlobs in container:")
for blob in container_client.list_blobs():
    print(f"{blob.name}: {blob.size} bytes")

# Step 5: Read Back
blob_client = container_client.get_blob_client(blob_path)

downloaded_bytes = blob_client.download_blob().readall()
downloaded_json = json.loads(downloaded_bytes)

os.makedirs("outputs", exist_ok=True)

with open("outputs/weather_raw.json", "w") as f:
    json.dump(downloaded_json, f, indent=2)

hourly_df = pd.DataFrame(downloaded_json["hourly"])

print("\nFirst 5 rows:")
print(hourly_df.head())