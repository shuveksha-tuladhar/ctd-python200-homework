# --- Part 2: Mini-Project -- Predicting Student Math Performance --- 
import pandas as pd
import matplotlib.pyplot as plt

# --- Task 1: Load and Explore --- 
df = pd.read_csv("student_performance_math.csv", sep=";")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)

# Histogram
plt.figure(figsize=(6, 5))
plt.hist(df["G3"], bins=21, color="green", edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Math Grades")
plt.ylabel("Frequency")
plt.savefig("outputs/g3_distribution.png")
plt.show()

# --- Task 2: Preprocess the Data --- 
print("Original shape:", df.shape)

df_filtered = df[df["G3"] != 0]
print("Filtered shape:", df_filtered.shape)

# Rows with G3=0 represent students who didn't take the final exam, not students who scored poorly. Keeping them would distort the model because their grade is not related to academic performance but rather absence from the exam.

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df_filtered[col] = df_filtered[col].map({"yes": 1, "no": 0})
                                            
df_filtered["sex"] = df_filtered["sex"].map({"F": 0, "M": 1})

corr_before = df["absences"].corr(df["G3"])
corr_after = df_filtered["absences"].corr(df_filtered["G3"])

print("Correlation (before filtering):", corr_before)
print("Correlation (after filtering):", corr_after)

# In the original dataset, many students with G3=0 had high absences because they didn't take the exam. This creates a misleading pattern where absences appear weakly related to grades.
# After filtering, we remove those "non-participants," revealing the true relationship: higher absences are more clearly associated with lower grades.

plt.scatter(df["absences"], df["G3"])
plt.title("Before Filtering")
plt.show()

plt.scatter(df_filtered["absences"], df_filtered["G3"])
plt.title("After Filtering")
plt.show()                                            