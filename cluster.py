import logging

from sklearn.cluster import Birch, KMeans
from sklearn.mixture import GaussianMixture


logger = logging.getLogger(__name__)


def cluster(data):
    inertias = []
    for i in range(1, len(data)):
        kmeans = build_clusters(data, i)
        inertias.append(kmeans.inertia_)
    return inertias


def build_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    # kmeans = Birch(threshold=threshold, n_clusters=n_clusters)
    # kmeans = GaussianMixture(n_components=n_clusters)
    kmeans.fit(data)
    return kmeans
