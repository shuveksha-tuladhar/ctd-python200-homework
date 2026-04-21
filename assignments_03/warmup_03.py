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

# --- KNN --- 
# KNN Q1 - Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), and predict on the test set. Print the accuracy score and the full classification report.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy (unscaled):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# KNN Q2 - Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). Print the accuracy score.

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("Accuracy (scaled):", accuracy_score(y_test, y_pred_scaled))
# Scaling often improves KNN because it relies on distance calculations,and features with larger scales can otherwise dominate the distance metric.For the Iris dataset, the improvement may be small because features are already on similar scales.

# KNN Q3 - Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. Print each fold score, the mean, and the standard deviation.

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("Fold scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())
# This result is more trustworthy than a single train/test split because it averages performance across multiple splits, reducing the impact of random variation.

# KNN Q4 - Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score.

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train, y_train, cv=5)
    print(f"k={k}, Mean CV Accuracy={scores.mean():.4f}")
    
# Select the value of k that produces the highest average cross-validation accuracy. 
# Smaller values like k=1 tend to memorize the training data, which can lead to overfitting and poor generalization to new data. 
# As k increases, the model becomes smoother and less sensitive to noise, improving generalization up to a point. However, if k becomes too large, the model may become overly simplistic and miss important patterns (underfitting).
# In practice, moderate values of k (often around 5–11) provide a good balance between these effects.

# --- Classifier Evaluation --- 
# Classifier Evaluation Q1 - Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, passing display_labels=iris.target_names.

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

# The model most often confuses versicolor and virginica (if any confusion occurs), since these two classes have more overlapping feature values compared to setosa.
