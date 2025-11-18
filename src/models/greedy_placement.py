import numpy as np


def greedy_diverse_selection(X, stations, k=5):
    """
    Greedy farthest-point selection.
    Picks stations that are maximally spread out in feature space.
    """
    X = np.array(X)
    n = X.shape[0]

    if n == 0:
        print("⚠ No stations provided.")
        return []
    if n <= k:
        print(f"⚠ Only {n} stations available. Returning all.")
        return stations

    norms = np.linalg.norm(X, axis=1)
    first_idx = int(np.argmax(norms))

    selected = [first_idx]
    remaining = set(range(n)) - set(selected)

    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    while len(selected) < k and remaining:
        min_dist_to_selected = {}
        for j in remaining:
            d_j = [dists[j, s] for s in selected]
            min_dist_to_selected[j] = min(d_j)

        next_idx = max(min_dist_to_selected, key=min_dist_to_selected.get)
        selected.append(next_idx)
        remaining.remove(next_idx)

    selected_stations = [stations[i] for i in selected]
    return selected_stations
