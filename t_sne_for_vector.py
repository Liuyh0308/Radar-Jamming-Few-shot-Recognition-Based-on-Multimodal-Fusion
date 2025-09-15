import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pickle

# Load features and labels
with open('./vector_with_maml_for_T-SNE/features.pickle', 'rb') as f:
    features = pickle.load(f)

with open('./vector_with_maml_for_T-SNE/labels.pickle', 'rb') as f:
    labels = pickle.load(f)

# Ensure features is a NumPy array
features = np.array(features)
labels = np.array(labels)

# Flatten each primary sample's sub-sample features
features_flat = features.reshape(features.shape[0], -1)
print("New feature shape after flattening:", features_flat.shape)

# Choose the label handling method
labels_first = labels[:, 0]

# Perform t-SNE with higher perplexity and learning rate, and more iterations
tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=500, n_iter=1000)
embedded_features = tsne.fit_transform(features_flat)

# # Perform K-means clustering on the t-SNE result
# num_clusters = len(np.unique(labels_first))
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# kmeans_labels = kmeans.fit_predict(embedded_features)

# Plot the result with clustering
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels_first, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('T-SNE Visualization with K-means Clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
