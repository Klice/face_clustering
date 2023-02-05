import logging

from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score


logger = logging.getLogger(__name__)


def cluster(data):
    inertias = []
    for i in range(2, len(data)):
        kmeans = build_clusters(data, i)
        score = silhouette_score(data, kmeans.labels_)
        inertias.append(score)
    return inertias


def build_clusters(data, n_clusters, threshold=0.1):
    # kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans = Birch(threshold=threshold, n_clusters=n_clusters)
    # kmeans = GaussianMixture(n_components=n_clusters)
    kmeans.fit(data)
    return kmeans
