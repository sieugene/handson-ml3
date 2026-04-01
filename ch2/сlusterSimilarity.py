from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # number of geographic clusters (regions)
        self.n_clusters = n_clusters
        
        # controls how fast similarity decreases with distance
        self.gamma = gamma
        
        # for reproducibility
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Step 1: find cluster centers using KMeans
        # X = [latitude, longitude]
        
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        
        # sample_weight makes expensive districts more important
        self.kmeans_.fit(X, sample_weight=sample_weight)
        
        return self

    def transform(self, X):
        # Step 2: compute similarity to each cluster center
        
        # output shape: (n_samples, n_clusters)
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]