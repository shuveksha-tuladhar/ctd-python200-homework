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


"""
1. This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?

This pipeline is very small, runs instantly, and has no external dependencies or failure points.
Using Prefect adds extra complexity (decorators, orchestration layer, dependency management) without providing meaningful benefits. A simple function call is easier to read, debug, and maintain
for such a lightweight task.

2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
Prefect becomes valuable when pipelines grow in complexity or operate in real-world environments.
- Processing large datasets where tasks may fail and need retries
- Workflows that depend on external systems (APIs, databases, cloud storage)
- Scheduled or recurring pipelines (e.g., daily data ingestion jobs)
- Pipelines requiring logging, monitoring, or alerting
- Parallel or distributed execution of tasks
- Managing dependencies between many steps in a data workflow

Even if each step is simple, Prefect helps coordinate, monitor, and scale the overall pipeline reliably.
"""