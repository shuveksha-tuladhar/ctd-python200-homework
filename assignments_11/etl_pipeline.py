# Part 2: Project -- Full ETL Pipeline

from dotenv import load_dotenv
from prefect import flow, task
from datetime import date
import json, os
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from openai import OpenAI

load_dotenv()

# setup
CITY = "Denver"

# Denver coordinates
LATITUDE = 39.7392
LONGITUDE = -104.9903

OPENAI_MODEL = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ACCOUNT_URL = "https://shuvekshactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&hourly=temperature_2m,precipitation"
    f"&forecast_days=7"
)

# --- Extract ---
@task(retries=2, retry_delay_seconds=10)
def extract_weather():
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    print(
        f"Successfully extracted weather data for {CITY}. "
        f"Retrieved {len(data['hourly']['time'])} hourly records."
    )

    return data

# --- Tranform ---
@task
def transform_weather(raw_data):
    client = OpenAI(api_key=OPENAI_API_KEY)

    hourly = raw_data["hourly"]

    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    precipitation = hourly["precipitation"]

    records = []

    for idx in range(len(times)):
        record = {
            "time": times[idx],
            "temperature_2m": temperatures[idx],
            "precipitation": precipitation[idx],
        }

        records.append(record)

    # Classify first 24 records
    for idx in range(min(24, len(records))):
        record = records[idx]

        system_prompt = (
            "You are classifying hourly weather conditions for outdoor running.\n"
            "Given a temperature in Celsius and a precipitation amount in mm,\n"
            "classify the conditions as exactly one of: good, marginal, or bad.\n"
            "Reply with that one word only -- no punctuation, no explanation."
        )

        user_prompt = (
            f"Temperature: {record['temperature_2m']} C\n"
            f"Precipitation: {record['precipitation']} mm"
        )

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            classification = (
                response.choices[0]
                .message.content.strip()
                .lower()
            )

            if classification not in {
                "good",
                "marginal",
                "bad",
            }:
                classification = "unknown"

        except Exception:
            classification = "unknown"

        record["running_condition"] = classification

        if (idx + 1) % 6 == 0:
            print(
                f"Classification progress: "
                f"{idx + 1}/24 records completed."
            )

    # Remaining records
    for idx in range(24, len(records)):
        records[idx]["running_condition"] = None

    print(
        f"Successfully transformed {len(records)} records."
    )

    return records

@task
def load_weather(records):
    today = date.today().isoformat()

    blob_path = f"final/{today}/weather_etl.json"

    payload = json.dumps(records, indent=2)

    credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential
    )

    container_client = blob_service_client.get_container_client(CONTAINER)

    container_client.upload_blob(
        name=blob_path,
        data=payload,
        overwrite=True,
    )

    byte_count = len(payload.encode("utf-8"))

    print(
        f"Uploaded blob to {blob_path} "
        f"({byte_count} bytes)"
    )

    return blob_path

@flow(log_prints=True)
def weather_etl_pipeline():
    raw_data = extract_weather()
    records = transform_weather(raw_data)
    blob_path = load_weather(records)

    print(
        f"Pipeline completed successfully. "
        f"Final blob path: {blob_path}"
    )

    return blob_path
    
if __name__ == "__main__":
    weather_etl_pipeline()