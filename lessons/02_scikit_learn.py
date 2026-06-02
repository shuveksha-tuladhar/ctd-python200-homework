from sklearn.linear_model import LinearRegression
import numpy as np

# temperature (degrees C) vs cupcakes sold
x_values = np.array([15, 18, 21, 24, 27]).reshape(-1, 1)
print(x_values.shape, x_values.ndim)
y_values = np.array([150, 200, 240, 310, 400])
new_x = np.array([17, 22]).reshape(-1, 1)

model = LinearRegression()                    # 1. create model
model.fit(x_values, y_values)                 # 2. fit model to data (learn)
y_predicted = model.predict(new_x)            # 3. predict with new data
print(y_predicted)  

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Synthetic student data with 3 natural groups.
# Values are on an arbitrary scale -- not real hours.
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42)
print(X.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: the raw, unlabeled data as the algorithm first sees it
ax1.scatter(X[:, 0], X[:, 1], color='gray', s=60, alpha=0.7)
ax1.set_title("Raw Data (No Labels)")
ax1.set_xlabel("Study Hours (synthetic scale)")
ax1.set_ylabel("Social Time (synthetic scale)")

# Right plot: what K-Means discovers
kmeans = KMeans(n_clusters=3, random_state=42)  # 1. Create the model
kmeans.fit(X)                                    # 2. Fit -- find cluster centers
labels = kmeans.predict(X)                       # 3. Predict a label for each point

ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
ax2.set_title("Student Clusters Found by K-Means")
ax2.set_xlabel("Study Hours (synthetic scale)")

plt.tight_layout()
plt.show()