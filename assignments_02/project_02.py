# --- Part 2: Mini-Project -- Predicting Student Math Performance --- 
import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Explore
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