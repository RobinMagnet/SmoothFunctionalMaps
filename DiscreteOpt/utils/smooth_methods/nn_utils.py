from sklearn.neighbors import NearestNeighbors
import numpy as np


def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : (n1,p) first collection
    Y : (n2,p) second collection
    k : int - number of neighbors to look for
    return_distance : whether to return the nearest neighbor distance
    use_ANN         : use Approximate Nearest Neighbors
    n_jobs          : number of parallel jobs. Set to -1 to use all processes

    Output
    -------------------------------
    dists   : (n2,k) or (n2,) if k=1 - ONLY if return_distance is False. Nearest neighbor distance.
    matches : (n2,k) or (n2,) if k=1 - nearest neighbor
    """

    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches
