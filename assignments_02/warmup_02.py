import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- scikit-learn API--- 
# scikit-learn Q1 : Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4,8]).reshape(-1,1)

model = LinearRegression() 
model.fit(years, salary)
salary_predicted = model.predict(new_years)

print("Slope (coef_):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", salary_predicted[0])
print("Predicted salary for 8 years:", salary_predicted[1])
 
# scikit-learn Q2 : Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.

x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

new_x = x.reshape(-1, 1)
print("New shape:", new_x.shape)

# scikit-learn expects X to be 2D because it treats the data as (samples, features). Even if there is only one feature, it still needs a column format so it knows how many data points (rows) and how many features (columns) there are.

# scikit-learn Q3 : Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels). Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels.

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
centers = kmeans.cluster_centers_
print("Cluster Centers:\n", centers)

cluster_counts = np.bincount(labels)
print("\nNumber of points in each cluster:", cluster_counts)

plt.figure(figsize=(6, 5))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200)

plt.title("K-Means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.savefig("outputs/kmeans_clusters.png")
plt.show()

# --- Linear Regression --- 

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Q1 : Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes.
plt.figure(figsize=(6, 5))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.show()

# The scatter plot shows a clear upward trend: medical cost increases with age. There are two distinct groups visible—one lower band (non-smokers) and one higher band (smokers).
# The smoker group is consistently shifted upward, suggesting that being a smoker significantly increases medical costs (by roughly a fixed amount).
# This indicates that the smoker variable has a strong impact on cost and is an important feature.

# Linear Regression Q2 : Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)

# Linear Regression Q3 : Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. Then predict on the test set.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("RMSE:", rmse)
print("R²:", r2)

# The slope represents how much medical cost increases for each additional year of age.

# Linear Regression Q4 : Add smoker as a second feature and fit a new model. Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:

X_full = np.column_stack([age, smoker])

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(Xf_train, yf_train)

r2_full = model_full.score(Xf_test, yf_test)

print("R² (age only):", r2)
print("R² (age + smoker):", r2_full)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient represents how much extra cost is added if a person is a smoker.