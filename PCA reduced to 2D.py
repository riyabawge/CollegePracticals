import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

score = silhouette_score(X_pca, clusters)

print("\nSilhouette Score:", score)

results = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Actual Class": y,
    "Cluster": clusters
})

print("\nCluster Comparison:")
print(pd.crosstab(results["Actual Class"], results["Cluster"]))

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    s=70
)
plt.title("K-Means Clustering on PCA Reduced Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results,
    x="PC1",
    y="PC2",
    hue="Actual Class",
    palette="Set2",
    s=70
)
plt.title("Actual Classes of Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()