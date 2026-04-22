import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


