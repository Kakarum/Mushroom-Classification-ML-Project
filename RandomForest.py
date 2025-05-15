import numpy as np
from DecisionTree import DecisionTree
import pandas as pd
from joblib import Parallel, delayed

# RandomForest class
class RandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', criterion='gini',
                 max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.n_estimators = n_estimators  # Number of trees in the forest
        self.trees = []
        self.max_features = max_features  # Number of features to consider when looking for the best split
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = None  # Total number of features
        self.feature_types = None

    def fit(self, X, y, feature_types):
        self.n_features = X.shape[1]
        self.feature_types = feature_types
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Train a decision tree
            tree = DecisionTree(criterion=self.criterion, max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_impurity_decrease=self.min_impurity_decrease, max_features=self.max_features)
            tree.fit(X_sample, y_sample, feature_types)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        # Majority voting
        predictions = np.array(predictions).T
        y_pred = [np.bincount(row).argmax() for row in predictions]
        return np.array(y_pred)