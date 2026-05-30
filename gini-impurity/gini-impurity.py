import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """

    def gini(y):
        y = np.asarray(y)

        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        return 1.0 - np.sum(probs ** 2)

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    return float(
        (n_left / n_total) * gini(y_left) +
        (n_right / n_total) * gini(y_right)
    )