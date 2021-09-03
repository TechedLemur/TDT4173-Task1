import numpy as np
import pandas as pd
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:

    def __init__(self):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.root = {}

    def fit(self, X, y):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement

        # Make a single dataframe, as it is easy to use Pandas functions on this to get the desired values
        data = X.join(y)
        attributes = X.columns.tolist()
        self.GOAL_ATTRIBUTE = y.name

        self.X = X
        #  The entropy of all the examples
        self.DATASET_ENTROPY = entropy(y.value_counts().values)

        # Number of values for the goal attribute
        self.S = np.sum(y.value_counts().values)

        self.root = self.recursive_fit(data, attributes)

    def recursive_fit(self, data, attributes, parent_examples=None):
        if data.empty:
            mode = parent_examples[self.GOAL_ATTRIBUTE].mode()
            node = {"value": random.choice(mode)}

            return node

        # Entropy is 0, we only have one type of goal attributes
        if (data[self.GOAL_ATTRIBUTE].value_counts().count() == 1):
            node = {"value": data[self.GOAL_ATTRIBUTE].value_counts().index[
                0]}
            return node

        A = ""
        val = 0.

        for a in attributes:

            grouped = data.groupby(a)[
                self.GOAL_ATTRIBUTE].value_counts().unstack(fill_value=0).stack()  # group data and fill in 0 if needed

            gain = self.DATASET_ENTROPY
            for sv in data[a].unique():
                gain -= float(np.sum(grouped[sv])) / self.S * \
                    entropy(grouped[sv].values)

            if gain >= val:
                A = a
                val = gain
        attributes.remove(A)
        node = {}
        node["label"] = A

        for v in self.X[A].unique():
            exs = data[data[A] == v]
            child = self.recursive_fit(exs, attributes.copy(), data)
            node[v] = child
        return node

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        # TODO: Implement

        predictions = []
        for index, row in X.iterrows():
            node = self.root
            finished = False
            while not finished:
                if "value" in node:
                    predictions.append(node["value"])
                    finished = True
                else:
                    attribute = row[node["label"]]
                    node = node[attribute]

        return np.array(predictions)

    def get_rules(self):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label

            attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


# --- Some utility functions

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a length k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))
