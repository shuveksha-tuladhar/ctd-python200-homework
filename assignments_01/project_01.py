from prefect import task, flow, get_run_logger
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
    
# Task 3: Visual Exploration
@task
def create_visualizations(df):
    logger = get_run_logger()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["Happiness score"], bins = 20, color="green", edgecolor="black")
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    
    output_file = output_dir / "happiness_histogram.png"
    plt.savefig(output_file)
    plt.close()
    logger.info(f"Histogram saved to: {output_file}")
    
    # Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="year", y="Happiness score", data=df)
    plt.title("Happiness Score Distribution by Year")

    boxplot_path = output_dir / "happiness_by_year.png"
    plt.savefig(boxplot_path)
    plt.close()
    logger.info(f"Boxplot saved to: {boxplot_path}")
    
    # Scatter: GDP vs Happiness
    plt.figure(figsize=(8, 5))
    plt.scatter(df["GDP per capita"], df["Happiness score"])
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")

    scatter_path = output_dir / "gdp_vs_happiness.png"
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f"Scatter plot saved to: {scatter_path}")
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))

    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 9})
    plt.title("Correlation Heatmap", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to: {heatmap_path}")
    
# Task 4: Hypothesis Testing
@task
def run_hypothesis_tests(df):
    logger = get_run_logger()
    alpha = 0.05

    # Test 1: 2019 vs 2020 (pandemic effect)
    scores_2019 = df.loc[df["year"] == 2019, "Happiness score"].dropna()
    scores_2020 = df.loc[df["year"] == 2020, "Happiness score"].dropna()

    t_stat_2019_2020, p_value_2019_2020 = stats.ttest_ind(
        scores_2019, scores_2020, equal_var=False
    )

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info("Hypothesis Test 1: 2019 vs 2020 Happiness Scores")
    logger.info(f"Mean 2019: {mean_2019:.3f}")
    logger.info(f"Mean 2020: {mean_2020:.3f}")
    logger.info(f"t-statistic: {t_stat_2019_2020:.4f}")
    logger.info(f"p-value: {p_value_2019_2020:.6f}")

    if p_value_2019_2020 < alpha:
        if mean_2020 > mean_2019:
            interpretation_2019_2020 = (
                "At alpha=0.05, the difference between 2019 and 2020 is statistically significant. "
                "In this dataset, average happiness was higher in 2020 than in 2019, suggesting a real "
                "shift rather than random variation."
            )
        else:
            interpretation_2019_2020 = (
                "At alpha=0.05, the difference between 2019 and 2020 is statistically significant. "
                "In this dataset, average happiness was lower in 2020 than in 2019, suggesting a real "
                "shift rather than random variation."
            )
    else:
        interpretation_2019_2020 = (
            "At alpha=0.05, the 2019 and 2020 averages are not statistically different. "
            "Any observed gap in this dataset could reasonably be due to chance."
        )

    logger.info(interpretation_2019_2020)

    # Test 2: North America and ANZ vs Sub-Saharan Africa
    region_a_name = "North America and ANZ"
    region_b_name = "Sub-Saharan Africa"

    region_a_scores = df.loc[
        df["Regional indicator"] == region_a_name, "Happiness score"
    ].dropna()
    region_b_scores = df.loc[
        df["Regional indicator"] == region_b_name, "Happiness score"
    ].dropna()

    t_stat_regions, p_value_regions = stats.ttest_ind(
        region_a_scores, region_b_scores, equal_var=False
    )

    mean_region_a = region_a_scores.mean()
    mean_region_b = region_b_scores.mean()

    logger.info(
        f"Hypothesis Test 2: {region_a_name} vs {region_b_name} Happiness Scores"
    )
    logger.info(f"Mean {region_a_name}: {mean_region_a:.3f}")
    logger.info(f"Mean {region_b_name}: {mean_region_b:.3f}")
    logger.info(f"t-statistic: {t_stat_regions:.4f}")
    logger.info(f"p-value: {p_value_regions:.6f}")

    if p_value_regions < alpha:
        higher_region = region_a_name if mean_region_a > mean_region_b else region_b_name
        interpretation_regions = (
            f"At alpha=0.05, the regional difference is statistically significant. "
            f"{higher_region} has a higher average happiness score in this dataset, and that gap is unlikely "
            "to be due to random sampling variation alone."
        )
    else:
        interpretation_regions = (
            "At alpha=0.05, these regional averages are not statistically different in this dataset. "
            "The observed gap could be due to chance."
        )

    logger.info(interpretation_regions)

    return {
        "test_2019_2020": {
            "mean_2019": mean_2019,
            "mean_2020": mean_2020,
            "t_statistic": t_stat_2019_2020,
            "p_value": p_value_2019_2020,
            "interpretation": interpretation_2019_2020,
        },
        "test_regions": {
            "region_a": region_a_name,
            "region_b": region_b_name,
            "mean_region_a": mean_region_a,
            "mean_region_b": mean_region_b,
            "t_statistic": t_stat_regions,
            "p_value": p_value_regions,
            "interpretation": interpretation_regions,
        },
    }


# Task 5: Correlation and Multiple Comparisons
@task
def run_correlation_tests(df):
    logger = get_run_logger()
    alpha = 0.05
    target_col = "Happiness score"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    explanatory_cols = [col for col in numeric_cols if col != target_col]

    number_of_tests = len(explanatory_cols)
    adjusted_alpha = alpha / number_of_tests

    logger.info("Task 5: Pearson Correlation Tests vs Happiness score")
    logger.info(f"Number of correlation tests: {number_of_tests}")
    logger.info(f"Original alpha: {alpha:.2f}")
    logger.info(f"Bonferroni adjusted alpha: {adjusted_alpha:.6f}")

    significant_alpha_0_05 = []
    significant_bonferroni = []
    correlation_results = {}

    for col in explanatory_cols:
        valid_df = df[[col, target_col]].dropna()

        if len(valid_df) < 3:
            logger.info(
                f"{col}: skipped (not enough non-missing paired observations)."
            )
            continue

        corr_coef, p_value = stats.pearsonr(valid_df[col], valid_df[target_col])
        is_sig_alpha = p_value < alpha
        is_sig_bonf = p_value < adjusted_alpha

        if is_sig_alpha:
            significant_alpha_0_05.append(col)
        if is_sig_bonf:
            significant_bonferroni.append(col)

        logger.info(
            f"{col}: r={corr_coef:.4f}, p-value={p_value:.6g}, "
            f"significant@0.05={is_sig_alpha}, significant@bonferroni={is_sig_bonf}"
        )

        correlation_results[col] = {
            "correlation_coefficient": corr_coef,
            "p_value": p_value,
            "significant_alpha_0_05": is_sig_alpha,
            "significant_bonferroni": is_sig_bonf,
        }

    logger.info(
        "Significant at alpha=0.05: "
        + (", ".join(significant_alpha_0_05) if significant_alpha_0_05 else "None")
    )
    logger.info(
        "Significant after Bonferroni correction: "
        + (", ".join(significant_bonferroni) if significant_bonferroni else "None")
    )

    return {
        "number_of_tests": number_of_tests,
        "adjusted_alpha": adjusted_alpha,
        "significant_alpha_0_05": significant_alpha_0_05,
        "significant_bonferroni": significant_bonferroni,
        "results": correlation_results,
    }

@flow
def happiness_pipeline():
    df = load_and_merge_data()
    compute_descriptive_stats(df)
    create_visualizations(df)
    run_hypothesis_tests(df)
    run_correlation_tests(df)
    
if __name__ == "__main__":
    happiness_pipeline()