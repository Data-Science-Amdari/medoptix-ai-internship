from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def fit_segmenter(X, n_clusters=3, pca_var=0.85, random_state=42):
    pca = PCA(n_components=pca_var, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_reduced)

    sil_score = silhouette_score(X_reduced, cluster_labels)
    print(f"✅ Silhouette Score: {sil_score:.3f}")

    return kmeans, pca, cluster_labels

def predict_cluster(X_new, model, reducer):
    X_new_reduced = reducer.transform(X_new)
    return model.predict(X_new_reduced)

def evaluate_clustering(X, labels, title="Patient Clustering"):
    tsne = TSNE(n_components=2, perplexity=40, init="pca", random_state=42)
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(label="Cluster")
    plt.show()
