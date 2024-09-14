import numpy as np

class MyRegModel:

    def __init__(self, n_iterations: int = 500, learning_rate: float = 0.01):
        """
        A simple model for Linear Regression predictions

        Args:
        n_iterations (int, default = 500): number of times the model will adjust weight and bias values
        learning_Rate (float, default = 0.01): value which will adjust the amount of change on bias and weight at iterations

        Methods:
        fit: Fit the model for a set of data
        predict: Predict value for a set of input
        """
        self.n_iters = n_iterations
        self.lr = learning_rate
        self.weight = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method intended for fitting the model to the data

        Args:
        X (np.ndarray): X matrix intended for training. e.g. [[Point 1 with n features], [Point 2 with n features], ..., [Point n with n features]]
        Y (np.ndarray): y array with training values. e.g. [y1 relative to Point 1, y2 relative to Point 2, ..., yn relative to Point n]
        """

        n_points, n_feats = X.shape
        
        self.weight = np.zeros(n_feats)
        self.bias = 0
        
        for _ in range(self.n_iters):

            y_pred = np.dot(X , self.weight) + self.bias

            self.weight -= self.lr * (2/n_points) * np.dot(X.T, (y_pred - y))
            self.bias -= self.lr * (2/n_points) * np.sum(y_pred - y)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the values based on calculated values (weight and bias) in fit method

        Args:
        X_test (np.ndarray): X matrix intended for testing. e.g. [[Point 1 with n features], [Point 2 with n features], ..., [Point n with n features]]

        Returns:
        np.ndarray with values predicted for every X_test point
        """
        return X_test*self.weight + self.bias