import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as py_stats
from scipy import stats

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

#%%
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

# %%
# --- NumPy ---
# NumPy Q1 - Create a 1D NumPy array from the list. Print its shape, dtype, and ndim.
new_arr = np.array([10, 20, 30, 40, 50])
print("Shape:", new_arr.shape)
print("Data type:", new_arr.dtype)
print("Dimensions:", new_arr.ndim)

# NumPy Q2 - Create the 2D array and print its shape and size (total number of elements).
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print("Shape:", arr.shape)
print("Total elements:", arr.size)

# NumPy Q3 - Using the 2D array from Q2, slice out the top-left 2x2 block and print it. 
print("Top-left 2x2 block:")
print(arr[:2, :2])

# NumPy Q4 - Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both. 
arr0 = np.zeros((3,4))
arr1 = np.ones((2,5))
print("Zero Array:","\n", arr0)
print("Ones Array:","\n", arr1)

# NumPy Q5 - Create an array using np.arange(0, 50, 5), print the array, its shape, mean, sum, and standard deviation
arr5 = np.arange(0, 50, 5)
print(arr5)
print("Array:", arr5)
print("Shape:", arr5.shape)
print("Mean:", np.mean(arr5))
print("Sum:", np.sum(arr5))
print("Standard Deviation:", np.std(arr5))

# NumPy Q6 - Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1. Print the mean and standard deviation of the result.
random_arr = np.random.normal(loc = 0, scale = 1, size = 200)
mean_val = np.mean(random_arr)
std_val = np.std(random_arr)

print(f"Mean: {mean_val}")
print(f"Standard Deviation: {std_val}")

# %%
# --- Matplotlib ---
# Matplotlib Q1 - Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2 - Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores, color="blue")
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3 - Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color="red", label="Dataset 1")
plt.scatter(x2, y2, color="green", label="Dataset 2")
plt.legend()
plt.title("Scatter Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

# Matplotlib Q4 - Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y)
axes[0].set_title("Line Plot (x vs y)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].bar(subjects, scores)
axes[1].set_title("Scores by Subject")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()
# %%
# --- Descriptive Stats ---
# Descriptive Stats Q1 - Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))  

# Descriptive Stats Q2 - Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
data_normal = np.random.normal(65, 10, 500)
plt.figure(figsize=(8, 5))
plt.hist(data_normal, bins = 20, color="green", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Score Range")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3 - Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Q4 - Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure(figsize=(8,5))

plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# The exponential distribution is more skewed (right-skewed).
# The median is a better measure of central tendency for the exponential data because it is less affected by extreme values.
# The normal distribution is symmetric, so the mean is an appropriate measure of central tendency.

# Descriptive Stats Q5 -  Print the mean, median, and mode. Why are the median and mean so different for data2?
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data1 Mean:", np.mean(data1))
print("Data1 Median:", np.median(data1))
print("Data1 Mode:", py_stats.mode(data1))

print("Data2 Mean:", np.mean(data2))
print("Data2 Median:", np.median(data2))
print("Data2 Mode:", py_stats.mode(data2))

# The mean and median are very different for data2 because of the extreme value (150), which is an outlier. The mean is heavily influenced by this large value and gets pulled upward, while the median remains more stable since it depends only on the middle value of the sorted data.

# %%
# --- Hypothesis ---
# Hypothesis Q1 - Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_statistic, p_value = stats.ttest_ind(group_a, group_b)

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.6f}")

# Hypothesis Q2 - Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")

# Hypothesis Q3 - Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# Hypothesis Q4 - Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, 70)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# Hypothesis Q5 - Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.
stats.ttest_ind(group_a, group_b, alternative="less")
print(f"One-tailed p-value: {p_val:.6f}")

# Hypothesis Q6 - Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("Students in group B scored higher on average than students in group A, and this difference is statistically significant, meaning it is unlikely to be due to random chance.")
# %%
