import numpy as np
from sklearn.cluster import KMeans


def kmeans_selection(X, stations, k=5, random_state=42):
    """
    Cluster stations and pick one representative (closest to centroid) per cluster.
    """
    X = np.array(X)
    n = X.shape[0]

    if n == 0:
        print("⚠ No stations provided.")
        return []
    if n <= k:
        print(f"⚠ Only {n} stations available. Returning all.")
        return stations

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    selected_indices = []

    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue

        cluster_points = X[idxs]
        centroid = centers[cluster_id]
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        best_local = idxs[np.argmin(dists)]
        selected_indices.append(best_local)

    selected_stations = [stations[i] for i in selected_indices]
    return selected_stations
