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

# Pandas Q2 - filter the rows to show only students who passed and have a grade above 80.   
filtered_df = df[(df["passed"]) & (df["grade"] > 80)]
print(f"Students who passed and have a grade above 80")
print(filtered_df)

# Pandas Q3 - Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).
df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4 - Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5 - Group the DataFrame by "city" and compute the mean grade for each city.
grouped = df.groupby("city")["grade"].mean()
print("Grouped Mean Grades by City:")
print(grouped)

# Pandas Q6 - Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7 - Sort the DataFrame by "grade" in descending order and print the top 3 rows.
sorted_df = df.sort_values("grade", ascending=False)
print(sorted_df.head(3))

