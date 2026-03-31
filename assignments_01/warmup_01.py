import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# --- Pandas ---
# Pandas Q1 - print the first three rows, the shape, and the data types of each column
print(f"First three rows")
print(df.head(3))
print(f"Shape of df: {df.shape}")

print(f"Data types of each column:")
print(df.dtypes)

# Print each result with a label
print(f"Num Columns: {len(df.columns)}")
print(f"Num Rows: {len(df)}")



