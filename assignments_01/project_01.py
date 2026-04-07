from prefect import task, flow, get_run_logger
import pandas as pd
import numpy as np
from pathlib import Path

# Task 1: Load Multiple Years of Data

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "happiness_project_csv"
    # Find all matching CSV files
    file_paths = sorted(data_dir.glob("world_happiness_*.csv"))

    if not file_paths:
        raise FileNotFoundError(f"No files found in: {data_dir}")

    dfs = []

    for path in file_paths:
        year = path.stem.split("_")[-1]
        logger.info(f"Loading file: {path}")

        df = pd.read_csv(
            path,
            sep=";",        
            decimal=",",
        )
        
        df["year"] = int(year)
        dfs.append(df)
        
    merged_df = pd.concat(dfs, ignore_index=True)

    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "merged_happiness.csv"
    merged_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved merged file to: {output_file}")
    logger.info(f"Final shape: {merged_df.shape}")
    
    return merged_df

# Task 2: Descriptive Statistics
@task
def compute_descriptive_stats(df):
    logger = get_run_logger()
    
    mean_score = df["Happiness score"].mean()
    median_score = df["Happiness score"].median()
    std_score = df["Happiness score"].std()
    
    yearly_mean = df.groupby("year")["Happiness score"].mean()
    
    regional_mean = df.groupby("Regional indicator")["Happiness score"].mean()
    
    logger.info(f"Overall Mean Happiness Score: {mean_score:.3f}")
    logger.info(f"Overall Median Happiness Score: {median_score:.3f}")
    logger.info(f"Overall Std Dev: {std_score:.3f}")
    logger.info("Mean Happiness Score by Year:")
    logger.info(f"\n{yearly_mean}")
    logger.info("Mean Happiness Score by Region:")
    logger.info(f"\n{regional_mean}")
    
    return {
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "yearly_mean": yearly_mean,
        "regional_mean": regional_mean
    }

@flow
def happiness_pipeline():
    df = load_and_merge_data()
    compute_descriptive_stats(df)
    
if __name__ == "__main__":
    happiness_pipeline()