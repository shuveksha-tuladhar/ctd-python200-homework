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

# --- The sklearn API: Decision Trees --- 
# Decision Trees Q1 - Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data, and predict on the test set. Print the accuracy score and classification report.

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# The Decision Tree accuracy is typically similar to or slightly lower than KNN on this dataset, though it can vary depending on the split.
# Since Decision Trees split based on feature thresholds rather than distance, scaling the data generally does not affect their performance.

# --- Logistic Regression and Regularization --- 
# Logistic Regression Q1 - Train three logistic regression models on the scaled Iris data, identical in every way except for the C parameter: C=0.01, C=1.0, and C=100. Use max_iter=1000 and solver='liblinear' for all three. For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum().

C_values = [0.01, 1.0, 100]

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    coef_magnitude = np.abs(model.coef_).sum()
    
    print(f"C={C}, Total |coefficients|={coef_magnitude:.4f}")
    
# As C increases, the total magnitude of the coefficients also increases. This is because a larger C means weaker regularization, allowing the model to fit the training data more closely with larger weights. 
# Conversely, a smaller C applies stronger regularization, shrinking coefficients toward zero to prevent overfitting.

# --- PCA ---
# PCA Q1 - Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), using cmap='gray_r' with each digit's label as the title.

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12, 3))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis('off')

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close() 

# PCA Q2 - Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits). Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar.

pca = PCA()
scores = pca.fit_transform(X_digits)

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap='tab10',
    s=10
)

plt.colorbar(scatter, label='Digit')
plt.title("PCA 2D Projection of Digits")

plt.savefig("outputs/pca_2d_projection.png")
plt.close()

# Same-digit images tend to form loose clusters in 2D PCA space, but there is still overlap because 2 components cannot fully separate all digit structure.

# PCA Q3 - Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using np.cumsum(pca.explained_variance_ratio_).

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.savefig("outputs/pca_variance_explained.png")
plt.close()

# About 20–30 components are typically needed to explain ~80% of the variance in the digits dataset.

# PCA Q4 - Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits using reconstruction through principal components n = 2, 5, 15, and 40.
# Build a grid of subplots where rows correspond to each n value and columns show those 5 digits. Add an "Original" row at the top (use images[i], which is already shaped as (8, 8))

def reconstruct_digit(sample_idx, scores, pca, n_components):
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction += scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
num_digits = 5
digit_indices = range(num_digits)

fig, axes = plt.subplots(len(n_values) + 1, num_digits, figsize=(10, 8))

# Original row
for j, idx in enumerate(digit_indices):
    axes[0, j].imshow(images[idx], cmap='gray_r')
    axes[0, j].set_title(f"Original {y_digits[idx]}")
    axes[0, j].axis('off')

# Reconstructions
for i, n in enumerate(n_values):
    for j, idx in enumerate(digit_indices):
        recon = reconstruct_digit(idx, scores, pca, n)
        axes[i + 1, j].imshow(recon, cmap='gray_r')
        axes[i + 1, j].set_title(f"n={n}")
        axes[i + 1, j].axis('off')

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()

# Digits usually become clearly recognizable around 15–20 components. This closely matches where the cumulative variance curve starts to plateau, showing that most structural information is captured early.