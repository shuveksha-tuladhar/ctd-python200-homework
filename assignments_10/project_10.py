# Reflection:
# Classifying weather conditions for outdoor running can be done with an LLM, but it is probably not the best use case. 
# A rule-based approach could classify conditions faster, cheaper, and more consistently. 
# An LLM may handle edge cases and nuanced judgments better, but for simple temperature and precipitation thresholds, deterministic code would likely be sufficient.

# Video link: https://youtu.be/8nTl39aQ5W8

from datetime import date
import json
import os

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ACCOUNT_URL = "https://shuvekshactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


def reshape_hourly(hourly):
    records = []

    for time, temp, precip in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"]
    ):
        records.append(
            {
                "time": time,
                "temperature_2m": temp,
                "precipitation": precip,
            }
        )

    return records


def main():
    credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential
    )

    container_client = blob_service_client.get_container_client(CONTAINER)

    # Step 1: Read
    today = "2026-06-02"  # use existing raw data instead of date.today()
    raw_blob_path = f"raw/{today}/weather.json"

    try:
        blob_client = container_client.get_blob_client(raw_blob_path)

        weather_data = json.loads(
            blob_client.download_blob().readall()
        )

        print(f"Loaded blob: {raw_blob_path}")

    except Exception:
        print("Using fallback dataset")

        with open(
            "assignments_10/resources/weather_raw.json",
            "r"
        ) as f:
            weather_data = json.load(f)

    records = reshape_hourly(weather_data["hourly"])

    # Only process first 24 records
    records = records[:24]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Step 2: Transform
    for index, record in enumerate(records, start=1):

        user_message = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        label = (
            response.choices[0]
            .message.content
            .strip()
            .lower()
        )

        if label not in VALID_LABELS:
            label = "unknown"

        record["conditions"] = label

        if index % 6 == 0:
            print(f"Processed {index} records")

    # Step 3: Write
    processed_blob_path = (
        f"processed/{today}/weather_classified.json"
    )

    container_client.upload_blob(
        name=processed_blob_path,
        data=json.dumps(records).encode("utf-8"),
        overwrite=True,
    )

    print(f"Uploaded {processed_blob_path}")

    # Step 4: Spot-Check
    processed_blob = (
        container_client
        .get_blob_client(processed_blob_path)
    )

    processed_records = json.loads(
        processed_blob.download_blob().readall()
    )

    df = pd.DataFrame(processed_records)

    print("\nCondition Counts:")
    print(df["conditions"].value_counts())

    print("\nFirst 5 Rows:")
    print(df.head())

    # Step 5: Save Output
    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/first_10_records.json",
        "w"
    ) as f:
        json.dump(
            processed_records[:10],
            f,
            indent=2,
        )

    print("\nSaved outputs/first_10_records.json")


if __name__ == "__main__":
    main()