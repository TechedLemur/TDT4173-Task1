import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self, alpha=0.1, iterations=200):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.alpha = alpha
        self.iterations = iterations

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        print("Training model, this might take a while...")
        o = np.ones((len(X), 1))
        x = X.copy(deep=True)  # Make a deep copy of X
        x["Bias"] = o  # add bias

        weights = np.ones(x.shape[1])  # initialize weights

        for i in range(self.iterations):
            for index, x_i in x.iterrows():
                # update weights with gradient ascent
                weights = weights + self.alpha * \
                    (y[index] - sigmoid(weights @ x_i)) * x_i

        self.weights = weights  # store weights in the class
        print("Finished training")

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """

        o = np.ones((len(X), 1))

        x = np.hstack((X, o))  # Add bias

        return sigmoid(x @ self.weights)


# --- Some utility functions

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss 

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))
