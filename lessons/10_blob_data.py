import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

ACCOUNT_URL = "https://shuvekshactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
VALID_LABELS = {"good", "marginal", "bad"}
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

# Read
today = date.today().isoformat()
credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

blob_name = f"raw/{today}/weather.json"
print(blob_name)

raw = container.download_blob(blob_name).readall()

# raw = container.download_blob(f"raw/{today}/weather.json").readall()
data = json.loads(raw.decode("utf-8"))

hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    records.append({
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    })

print(f"Loaded {len(records)} records")

# Transform
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 24 == 0:
        print(f"  Processed {i + 1} records...")

# Load
processed_path = f"processed/{today}/weather_classified.json"
payload = json.dumps(enriched).encode("utf-8")
container.upload_blob(processed_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {processed_path}")

# Spot-check
df = pd.DataFrame(enriched)
print("\nLabel distribution:")
print(df["conditions"].value_counts())
print(df.head(10))