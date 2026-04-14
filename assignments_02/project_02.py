# --- Part 2: Mini-Project -- Predicting Student Math Performance --- 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# --- Task 3: Exploratory Data Analysis ---
numeric_cols = df_filtered.select_dtypes(include=["int64", "float64"])
corr_with_g3 = numeric_cols.corr()["G3"].sort_values()
print("Correlation with G3 (sorted):\n")
print(corr_with_g3)

# The strongest relationship with G3 is G2 (and also G1), which makes sense because earlier grades are strong predictors of the final grade.
# The strongest negative relationships are failures and absences. This shows that past academic performance is the most important predictor, while behavioral factors have weaker effects.

# Visualization 1: G2 vs G3
plt.figure(figsize=(6, 5))
plt.scatter(df_filtered["G2"], df_filtered["G3"])
plt.title("G2 vs Final Grade (G3)")
plt.xlabel("G2 (Second Period Grade)")
plt.ylabel("G3 (Final Grade)")
plt.savefig("outputs/g2_vs_g3.png")
plt.show()
# There is a very strong linear relationship between G2 and G3. Points cluster along an upward line, meaning students who do well earlier tend to do well on the final. G2 is a very strong predictor.

# Visualization 2: Absences vs G3
plt.figure(figsize=(6, 5))
plt.scatter(df_filtered["absences"], df_filtered["G3"])
plt.title("Absences vs Final Grade (G3)")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/absences_vs_g3.png")
plt.show()
# There is a slight negative relationship between absences and G3. Students with more absences tend to have somewhat lower grades, but the relationship is noisy. This suggests attendance matters, but it is not as strong a predictor as prior grades (G1, G2).

# Visualization 3: Failures vs G3
plt.figure(figsize=(6, 5))
plt.scatter(df_filtered["failures"], df_filtered["G3"])
plt.title("Failures vs Final Grade (G3)")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/failures_vs_g3.png")
plt.show()
# There is a negative relationship between failures and G3. Students with more past failures tend to have lower final grades. High-performing students almost always have few or no past failures.

# --- Task 4: Baseline Model ---

X = df_filtered[["failures"]]   
y = df_filtered["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("slope (failures):", model.coef_[0])
print("intercept:", model.intercept_)
print("RMSE:", rmse)
print("R^2:", r2)

# The slope is negative, meaning each additional past failure lowers the predicted final grade. Since grades are on a 0–20 scale, the slope tells us how many grade points are lost per failure.
# The RMSE shows the typical prediction error in grade points. If RMSE is around 2–4, that means predictions are off by a few points on average, which is fairly large relative to the 0–20 grading scale.
# The R^2 is relatively low, which makes sense because we are only using one weak/moderate predictor (failures). This model is much worse than models that would include G1 or G2, which had much stronger correlations.

# --- Task 5: Build the Full Model ---
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_filtered[feature_cols].values
y = df_filtered["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("Train R^2:", train_r2)
print("Test R^2:", test_r2)
print("RMSE:", rmse)

# The test R^2 here should be noticeably higher than the baseline model using only failures, showing that adding more features improves predictive performance.

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
    
# Coefficient Analysis 

# Sorted roughly by absolute impact (largest to smallest):
# schoolsup   : -2.062   <-- strongest effect
# failures    : -1.145
# internet    : +0.834
# higher      : +0.610
# sex         : +0.453
# studytime   : +0.448
# Fedu        : +0.186
# traveltime  : -0.112
# Medu        : +0.083
# freetime    : -0.042
# activities  : -0.009

# Interpretations

# 1. schoolsup (-2.062) is surprisingly NEGATIVE and very large.
#    Students receiving school support tend to have LOWER final grades. This is likely NOT causal — it's selection bias. Students who are already struggling are the ones who get extra support, so "schoolsup" is acting as a proxy for "struggling student."

# 2. internet (+0.834) is positive and relatively strong.
#    Access to internet likely helps with studying/resources, so this makes intuitive sense.

# 3. higher (+0.610) is positive.
#    Students who want to pursue higher education tend to perform better. This aligns well with expectations (motivation effect).

# 4. sex (+0.453) is moderately positive.
#    Since sex was encoded (likely M=1, F=0), this suggests one group performs higher.This could reflect dataset-specific patterns rather than a general rule.

# 5. activities (-0.009) is essentially zero (no effect).
#    Participation in activities doesn't meaningfully impact grades here.

# 6. freetime (-0.042) is slightly negative.
#    More free time might slightly reduce study time, but effect is very small.

# The model generalizes well (train and test R² are close), but overall performance is weak, explaining only ~15% of grade variation. Predictions are off by about ±3 points on average, which is fairly large on a 0–20 scale. 
# Adding features helped slightly over the baseline, but the model is still limited due to missing stronger predictors like G1 and G2.

# For a production model, it is advisable to retain features that show meaningful predictive power and clear real-world interpretation, such as failures, studytime, higher education aspirations, and parental education levels (Medu, Fedu).
# Features like freetime and activities contribute little to model performance and may introduce noise, while variables such as internet access and sex should be carefully evaluated due to their weak impact and potential ethical considerations.
# Adding multiple features improves performance, but many features contribute only small amounts. A simpler model with the strongest predictors may be more robust and interpretable.
