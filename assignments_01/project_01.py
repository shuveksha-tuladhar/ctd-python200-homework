from prefect import task, flow, get_run_logger
import pandas as pd
from pathlib import Path

# Task 1: Load Multiple Years of Data

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    data_dir = Path("happiness_project")
    # Find all matching CSV files
    file_paths = sorted(data_dir.glob("world_happiness_*.csv"))

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

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "merged_happiness.csv"
    merged_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved merged file to: {output_file}")
    logger.info(f"Final shape: {merged_df.shape}")
    
    return merged_df

@flow
def happiness_pipeline():
    df = load_and_merge_data()
    
if __name__ == "__main__":
    happiness_pipeline()