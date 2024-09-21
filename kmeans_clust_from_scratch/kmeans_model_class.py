import numpy as np

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float | np.ndarray:
    """
    Function responsible to calculate the euclidean distance between 2 arrays.
    Each array represents a point, where every item is a coordinate in a dimension.

    Args:
    p1 (np.ndarray): array of coordinates 1
    p2 (np.ndarray): array of coordinates 2. This array may be a array of points too.

    Returns:
    Euclidean Distance between points p1 and p2 (float)
    If p2 is an array of points, then the return will be a array representing the distances from p1 to each point in array p2
    """
    return np.sqrt(np.sum((p1-p2)**2, axis=len(p2.shape)-1))

class MyKMeansModel:

    def __init__(self, K: int = 3, max_iters: int = 10, plot_iters: bool = False):
        """
        Class for K Means Clusterization

        Args:
        K (int): number of cluster
        max_iters (int): max number of iterations for clustering
        plot_iters (bool): whether ploting eac iter or not
        """
        self.K = K
        self.max_iters = max_iters
        self.plot_iters = plot_iters

    def _update_clusters(self, centroids: np.ndarray) -> list[list]:
        """
        Helper function to update or create clusters based on euclidean distance of the points to centroids

        args:
        centroids (np.ndarray): array with centroids coordinates

        return:
        return list of clusters (which are lists of points)
        """

        # initialize empty clusters
        clusters = [[] for _ in range(self.K)]

        for sample in self.X:

            # index of min distance from sample to centroids 
            sample_cluster_index = np.argmin(euclidean_distance(sample,centroids))
            clusters[sample_cluster_index].append(sample)

        return clusters

    def _update_centroids(self, clusters: list, n_features: int) -> np.ndarray:
        """
        Helper function to update centroids values based on mean values of clusters

        args:
        clusters (list): list of clusters (which are lists of points)

        return:
        array of coords for each centroid
        """
        
        # initialzie empty centroids. Since each centroid is a point, is should have the same shape as the sample points
        centroids = np.zeros((self.K, n_features))

        # calculate the mean for each cluster and append to centroid
        for idx, _ in enumerate(clusters):
            cluster_mean = np.mean(clusters[idx], axis=0)
            centroids[idx] = cluster_mean
        
        return centroids

    def _is_converged(self, centroids_old: np.ndarray, centroids: np.ndarray) -> bool:
        """
        Helper function to verify if centroids location have changed since last iteration

        args:
        centroids_old (np.ndarray): array with old centroids coordinates
        centroids (np.ndarray): array with centroids coordinates

        return:
        wheter the centroids lcoation have changed or not

        """
        return np.sum(euclidean_distance(centroids_old, centroids)) == 0
    
    def _get_labels(self, clusters: list, n_samples: int) -> np.ndarray:
        """
        Helper function to associate a label (cluster) to each point

        args:
        clusters (list): list fo clusters (which are lists of points)
        n_samples (int): number of samples (points) in the original dataset  

        return:
        array with labels for each point
        """

        # initialize empty labels array
        labels = np.zeros(n_samples)


        for cluster_index, cluster in enumerate(clusters):
            for sample in cluster:
                
                # append label (cluster index) at the sample position 
                labels[np.where(sample == self.X)[0][0]] = cluster_index

        return np.array(labels) 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method for data clustering

        args:
        X (np.ndarray): dataset pro clustering

        returns:
        array with points cluster identification
        """

        self.X = X
        n_samples, n_features = X.shape

        # initliaze random centroids
        initial_centroids_index = np.random.choice(n_samples, self.K, replace=False) 
        centroids = X[initial_centroids_index]

        # plotting
        if self.plot_iters and self.K <=5:
            import matplotlib.pyplot as plt
            colors = ['red', 'limegreen', 'tab:blue', 'purple', 'c']

        for i in range(self.max_iters):

            clusters = self._update_clusters(centroids)
            
            centroids_old = centroids
            centroids = self._update_centroids(clusters, n_features)

            # plotting
            if self.plot_iters and self.K <=5:
                _, ax = plt.subplots()
                ax.scatter(X[:,0], X[:,1], color=[colors[int(item)] for item in self._get_labels(clusters, n_samples)], marker='.')
                ax.scatter(np.array(centroids_old)[:,0], np.array(centroids_old)[:,1], marker='x', color='k')

            if self._is_converged(centroids_old, centroids): break
            if i == range(self.max_iters)[-1]: print('Last Iteration')


        return self._get_labels(clusters, n_samples)