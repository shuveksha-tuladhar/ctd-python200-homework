import pandas as pd
import numpy as np
from prefect import task, flow

# --- Pipelines ---
# Pipelines Q2 - Prefect

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Step 1: Create Series
@task
def create_series(arr):
    return pd.Series(arr, name="values")

# Step 2: Clean data
@task
def clean_data(series):
   return series.dropna()

# Step 3: Summarize data
@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return summary

if __name__ == "__main__":
    pipeline_flow()
