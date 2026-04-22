import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

# Mini-Project -- Spam or Ham? A Classifier Shootout

# Task 1: Load and Explore

# Load dataset

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
df.head()

# Basic exploration
print("\nDataset shape:")
print(df.shape)

print("\nClass counts:")
print(df["spam_label"].value_counts())

print("\nClass proportions:")
print(df["spam_label"].value_counts(normalize=True))

# Boxplots for key features

features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features:
    plt.figure(figsize=(6, 4))

    # Separate spam vs ham
    spam = df[df["spam_label"] == 1][feature]
    ham = df[df["spam_label"] == 0][feature]

    plt.boxplot([ham, spam], labels=["Ham (0)", "Spam (1)"])
    plt.title(f"{feature} by Spam vs Ham")
    plt.ylabel(feature)

    filename = f"outputs/{feature}_boxplot.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")

print("\nSummary statistics (selected features):")
print(df[features].describe())

# The dataset has 4,601 emails with a moderate class imbalance (~61% ham, ~39% spam). This means accuracy alone can be misleading, since always predicting “ham” gives ~60%.
# Boxplots show spam tends to have higher values for all three features. The differences are subtle for word_freq_free and char_freq_!, but more dramatic for capital_run_length_total.
# Many features are mostly zero, indicating sparse data where most emails don’t contain most words. Feature scales vary widely (small frequencies vs large counts), which matters for scale-sensitive models like logistic regression, KNN, and SVM.

# Task 2: Prepare Your Data

# Train/test split
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

# Use stratify to preserve class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA (fit on training only)
pca = PCA()
pca.fit(X_train_scaled)

# Cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)

# Find number of components to reach 90%
n = np.argmax(cum_var >= 0.90) + 1
print(f"\nNumber of components for 90% variance: {n}")

# Plot cumulative variance
plt.figure(figsize=(6, 4))
plt.plot(cum_var)
plt.axhline(0.90, linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.savefig("outputs/pca_explained_variance.png", bbox_inches="tight")
plt.close()

# Transform data using PCA
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# We split the data into training and testing sets using stratification to preserve the class balance. This ensures both sets reflect the original distribution of spam and ham.
# We scaled the features using StandardScaler because the dataset contains features with very different magnitudes (small frequencies vs large counts).
# Scaling is important for models like KNN and logistic regression.
# PCA was applied after scaling, since it is sensitive to feature magnitude. It was fit only on the training data to avoid data leakage.
# We selected the number of components that explain at least 90% of the variance, then transformed both training and test sets. We keep both the scaled data and PCA-reduced data for use in different models.

# Task 3: A Classifier Comparison

# KNN (unscaled)
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred = knn_unscaled.predict(X_test)

print("\nKNN (unscaled)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# KNN (scaled)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN (scaled)")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))
print(classification_report(y_test, y_pred_scaled))

# KNN (PCA)
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)

print("\nKNN (PCA)")
print("Accuracy:", accuracy_score(y_test, y_pred_pca))
print(classification_report(y_test, y_pred_pca))

# Decision Tree (depth tuning)
depths = [3, 5, 10, None]

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    print(f"\nDecision Tree (max_depth={d})")
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

tree_final = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_final.fit(X_train, y_train)
y_pred_tree = tree_final.predict(X_test)

print("\nDecision Tree (final)")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Logistic Regression (scaled)
log_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_scaled.fit(X_train_scaled, y_train)
y_pred_log_scaled = log_scaled.predict(X_test_scaled)

print("\nLogistic Regression (scaled)")
print("Accuracy:", accuracy_score(y_test, y_pred_log_scaled))
print(classification_report(y_test, y_pred_log_scaled))

# Logistic Regression (PCA)
log_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_pca.fit(X_train_pca, y_train)
y_pred_log_pca = log_pca.predict(X_test_pca)

print("\nLogistic Regression (PCA)")
print("Accuracy:", accuracy_score(y_test, y_pred_log_pca))
print(classification_report(y_test, y_pred_log_pca))

# Confusion Matrix (best model)
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Best Model Confusion Matrix")
plt.savefig("outputs/best_model_confusion_matrix.png", bbox_inches="tight")
plt.close()

# KNN performs poorly on unscaled data (~0.80 accuracy) because large-scale features dominate distance calculations. After scaling, performance improves significantly (~0.91). PCA gives nearly identical performance, suggesting dimensionality reduction does not add much benefit here.
# For the decision tree, training accuracy increases steadily with depth, reaching nearly perfect accuracy at max_depth=None, while test accuracy improves only slightly. This indicates overfitting at higher depths. A moderate depth like 5 provides a good balance between performance and generalization.
# Random Forest performs best (~0.95 accuracy), likely because it reduces overfitting and captures complex patterns through ensembling.
# Logistic Regression performs well on scaled data (~0.93), but PCA slightly reduces performance (~0.92), indicating some useful information is lost in dimensionality reduction. This matches expectations from Task 2.
# Overall, Random Forest is the best-performing model. PCA did not significantly improve results for KNN or Logistic Regression, which aligns with the hypothesis that some information is lost during dimensionality reduction.
# For a spam filter, accuracy is not the most important metric. False positives(legitimate emails marked as spam) are more costly than false negatives, so minimizing false positives is more important.
# The best model (Random Forest) makes slightly more false negatives than false positives (spam missed vs legitimate emails blocked), which is preferable since it avoids incorrectly filtering legitimate emails.

# Feature importances
feature_names = X.columns

# Decision Tree importances
tree_importances = tree_final.feature_importances_
tree_top_idx = tree_importances.argsort()[::-1][:10]

print("\nTop 10 Decision Tree Features:")
for i in tree_top_idx:
    print(feature_names[i], tree_importances[i])

# Random Forest importances
rf_importances = rf.feature_importances_
rf_top_idx = rf_importances.argsort()[::-1][:10]

print("\nTop 10 Random Forest Features:")
for i in rf_top_idx:
    print(feature_names[i], rf_importances[i])

# Plot Random Forest importances
top_features = [feature_names[i] for i in rf_top_idx]
top_values = rf_importances[rf_top_idx]

plt.figure(figsize=(8, 5))
plt.barh(top_features[::-1], top_values[::-1])
plt.xlabel("Importance")
plt.title("Top 10 Random Forest Feature Importances")
plt.savefig("outputs/feature_importances.png", bbox_inches="tight")
plt.close()

# The Decision Tree and Random Forest both identify similar important features, especially char_freq_$, char_freq_!, word_freq_remove, word_freq_free, and capital run length features.
# The Decision Tree is heavily influenced by a few dominant features (especially char_freq_$), while the Random Forest spreads importance more evenly across multiple features, including capital_run_length statistics and common words like "your" and "you".
# Overall, both models align with intuition: spam emails are strongly associated with dollar signs, exclamation marks, removal requests, and words like "free", as well as excessive capitalization.
# The Random Forest provides a more stable and realistic view of feature importance because it averages across many trees, reducing the risk of overfitting that affects a single Decision Tree.

# Task 4: Cross-Validation

models = {
    "KNN (scaled)": knn_scaled,
    "Decision Tree": tree_final,
    "Random Forest": rf,
    "Logistic Regression (scaled)": log_scaled
}

print("\nCross Validation Results (cv=5)")

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled if "scaled" in name else X_train, y_train, cv=5)
    print(f"\n{name}")
    print("Mean Accuracy:", scores.mean())
    print("Std Dev:", scores.std())

# Cross-validation shows Random Forest performs best overall, with the highest mean accuracy (~0.954). Logistic Regression is the second-best model (~0.924), followed by Decision Tree (~0.907) and KNN (~0.905).
# In terms of stability, KNN and Logistic Regression have the lowest standard deviation (~0.009), meaning their performance is very consistent across folds. Random Forest is slightly more variable but still stable given its higher accuracy.
# The Decision Tree shows the highest variance, which reflects its sensitivity to training data splits and tendency to overfit compared to ensemble methods.
# Overall, the cross-validation results match the single train/test split ranking: Random Forest is best, followed by Logistic Regression, then Decision Tree and KNN. This confirms that the earlier results were not due to a lucky or unlucky split.