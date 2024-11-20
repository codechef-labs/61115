import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import time
from datetime import datetime

current_time = int(time.time())  # Convert to milliseconds
np.random.seed(current_time)


# Generate synthetic data
n_samples = 300
n_features = 2
n_clusters = 3
cluster_std = 2

# Create synthetic data points
X, y = make_blobs(n_samples=n_samples,
                  n_features=n_features,
                  centers=n_clusters,
                  cluster_std=cluster_std,
                  random_state=42)

# Create and fit KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Create visualization
plt.figure(figsize=(10, 6))

# Plot the data points with their assigned clusters
for i in range(n_clusters):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], 
               cluster_points[:, 1], 
               label=f'Cluster {i}',
               alpha=0.6)

# Plot the cluster centers
plt.scatter(centers[:, 0], 
           centers[:, 1], 
           c='black',
           marker='x',
           s=200,
           linewidths=3,
           label='Centroids')

plt.title('K-means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()

# Print additional information
print("\nCluster Information:")
for i in range(n_clusters):
    cluster_size = np.sum(labels == i)
    print(f"Cluster {i}: {cluster_size} points")

# Calculate and print the inertia (within-cluster sum of squares)
print(f"\nInertia: {kmeans.inertia_:.2f}")
