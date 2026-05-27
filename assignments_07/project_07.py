# --- Part 2: Mini-Project — World Happiness Agent ---
from dotenv import load_dotenv
from openai import OpenAI
import os
import matplotlib
# Force a non-interactive backend before pyplot is ever imported. The agent's
# plotting code runs in a worker thread, and an interactive GUI backend (e.g.
# the macOS backend) crashes when it tries to open a window off the main
# thread ("NSWindow should only be instantiated on the main thread!").
matplotlib.use("Agg")
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent, OpenAIServerModel

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
print('OpenAI client created.')

# Pre-task: Load the Data
df = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_PATH = PROJECT_ROOT / "assignments_01" / "outputs" / "merged_happiness.csv"
FALLBACK_DIR = PROJECT_ROOT / "assignments" / "resources" / "happiness_project"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Task 1: Define the Tools
# Tool 1: load_happiness_data
@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.

    Loads the CSV from DATA_PATH. If that file does not exist,
    loads and merges all yearly CSVs from assignments/resources/happiness_project/.

    Column names are normalized to snake_case (e.g. "Happiness score" ->
    "happiness_score", "Regional indicator" -> "region") and the per-year
    score columns are coalesced into a single happiness_score column.
    Stores the result in the global df.

    Returns:
        dict: A dictionary containing:
            - "shape": the (rows, columns) tuple of the loaded DataFrame.
            - "columns": the list of normalized column names.
    """

    global df

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        frames = []

        for year in range(2015, 2025):
            path = os.path.join(FALLBACK_DIR, f"{year}.csv")

            if os.path.exists(path):
                yearly = pd.read_csv(path)
                yearly["year"] = year
                frames.append(yearly)

        if not frames:
            return {"error": "No happiness data found."}

        df = pd.concat(frames, ignore_index=True)

    rename_map = {
        "Ranking": "ranking",
        "Country": "country",
        "Regional indicator": "region",
        "Happiness score": "happiness_score",
        "GDP per capita": "gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy": "healthy_life_expectancy",
        "Freedom to make life choices": "freedom",
        "Generosity": "generosity",
        "Perceptions of corruption": "perceptions_of_corruption",
        "Ladder score": "ladder_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "ladder_score" in df.columns and "happiness_score" in df.columns:
        df["happiness_score"] = df["happiness_score"].fillna(df["ladder_score"])

    return {
        "shape": df.shape,
        "columns": list(df.columns),
    }

# Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
        Uses df[column].describe().to_dict().
        Returns {"error": "..."} if no data is loaded or column is missing.

    Args:
        column (str): The name of the column to summarize.
    """
    global df
    
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    return df[column].describe().to_dict()
    
# Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
     Uses scipy.stats.pearsonr to calculate the linear relationship
    between two numeric columns.

    Args:
        col1 (str): The first numeric column.
        col2 (str): The second numeric column.

    Returns:
        dict: A dictionary containing:
            - "col1"
            - "col2"
            - "pearson_r"
            - "p_value"

        Returns {"error": "..."} if:
            - no data is loaded
            - either column is missing
            - correlation cannot be computed
    """

    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found."}

    try:
        clean_df = df[[col1, col2]].dropna()

        r, p = pearsonr(clean_df[col1], clean_df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }

    except Exception as e:
        return {"error": str(e)}

# Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the loaded dataset to the requested year, sorts by the given
    column in descending order, and returns the highest-ranking rows.

    Args:
        column (str): The name of the column to rank countries by.
        year (int): The year to filter the dataset on.
        n (int): The number of top countries to return. Defaults to 5.

    Returns:
        list: A list of dicts, one per country, each containing "country"
            and the value of the requested column. Returns {"error": "..."}
            if no data is loaded, the column/year is missing, or the year
            has no rows.
    """
    global df
    if df is None:
        return {"error": "No data loaded."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if "year" not in df.columns:
        return {"error": "Dataset has no 'year' column."}

    subset = df[df["year"] == year]
    if subset.empty:
        return {"error": f"No data for year {year}."}

    top = subset.sort_values(column, ascending=False).head(n)
    return [
        {"country": str(row["country"]), column: float(row[column])}
        for _, row in top.iterrows()
    ]

# Task 2: Build the Agent

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

The full cleaned dataset is already available in your Python environment as a
pandas DataFrame named `df`. Its columns are normalized to snake_case:
country, region, happiness_score, gdp_per_capita, social_support,
healthy_life_expectancy, freedom, generosity, perceptions_of_corruption,
year, ranking. For any custom analysis or plotting, use `df` directly.

Important: load_happiness_data() returns only a summary dict with the keys
"shape" and "columns" - it does NOT return the DataFrame. Never call DataFrame
methods such as .shape, .groupby(), or .head() on the value it returns; use the
`df` variable instead.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# Task 3: Run Guided Queries
if __name__ == "__main__":
    os.chdir(BASE_DIR)

    load_happiness_data()
    agent.state["df"] = df

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

# Task 4 Queries
    # My query 1
    my_query_1 = (
        "Which region has the highest average happiness_score across all years?"
    )

    response_1 = agent.run(my_query_1, reset=False)

    print("\n--- My Query 1 ---")
    print(response_1)

    # This primarily triggered tool use plus some generated analysis code.

    # My query 2
    my_query_2 = (
        "Create a histogram of happiness_score and save it to "
        "outputs/happiness_histogram.png."
    )

    response_2 = agent.run(my_query_2, reset=False)

    print("\n--- My Query 2 ---")
    print(response_2)

    # This triggered code generation because no plotting tool exists.

# --- Reflection ---
# 1. In Query 3, how did the agent communicate whether the correlation was statistically significant? Did it use the p-value correctly? What threshold did it apply?
#    The agent used the p-value returned from the Pearson correlation calculation to determine statistical significance. It correctly interpreted a p-value below 0.05 as statistically significant, which is the standard threshold commonly used in data analysis.

# 2. Did any of the agent's responses surprise you — either by being more capable than you expected, or less? Describe one specificexample.
#    One surprising capability was the agent's ability to generate custom matplotlib plotting code automatically when the available tools were insufficient. 
#    For the regional line chart query, it correctly grouped the data, created the visualization, and saved the figure without requiring a dedicated plotting tool.

# 3. What one additional tool would make this agent meaningfully more useful?
#    A useful additional tool would be: compare_country_trends(country1, country2)
#    This tool could compare happiness metrics between two countries over time and return summary statistics or trend data. 
#    It would help answer questions such as: "How has happiness changed in Canada versus the United States over the past decade?"