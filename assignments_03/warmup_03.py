import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing --- 
# Preprocessing Q1 - Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42. Print the shapes of all four arrays.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preprocessing Q2 - Fit a StandardScaler on X_train and use it to transform both X_train and X_test. Print the mean of each column in X_train_scaled -- they should all be very close to 0. Add a comment explaining in one sentence why you fit the scaler on X_train only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test) 
print("Column means of X_train_scaled:", X_train_scaled.mean(axis=0))

# The scaler is fit only on X_train to avoid data leakage, ensuring that information from the test set does not influence the model.