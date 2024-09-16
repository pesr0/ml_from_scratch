import numpy as np
from collections import Counter

def euclidean_distance(X_train: np.ndarray, X_test: np.ndarray) -> float:
    """
    Function responsible to calculate the euclidean distance between 2 arrays.
    Each array represents a point, where every item is a coordinate in a dimension.

    Args:
    X_train (np.ndarray): array of coordinates 1
    X_test (np.ndarray): array of coordinates 2

    Retruns:
    Euclidean Distance between points X_train and X_test (float)
    """
    return np.sqrt(np.sum((X_train-X_test)**2))

class MyKnnModel:

    def __init__(self, k: int= 3):
        """
        KNN class initiantor

        Args:
        k (int, default = 3): number of K neighbors to consider
        """
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method

        Args:
        X (np.ndarray): Array with points coordinates for training
        y (np.ndarray): Array with labels for each point in X 
        """
        self.X = X
        self.y = y

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict method, this method will return the predictions for the inputed data points

        Args:
        X_test (np.ndarray): Array with point(s) cordinates

        Returns:
        Array with predctions for each point in X_test 
        """
        
        # creates a empty array for labels returning
        return_lables = np.array([])

        # iterate over each point in X_test
        for x in X_test:

            # creates a list with every euclidean distance between inserted point and every training point
            distances = [euclidean_distance(x, point) for point in self.X]
            
            # order and gat the most repetitive label betwenn k neighbors
            k_index = np.argsort(distances)[:self.k]
            labels = self.y[k_index]
            most_common = Counter(labels).most_common()[0][0]

            # append mot common label to return_labels
            return_lables = np.append(return_lables, [most_common])

        return return_lables






