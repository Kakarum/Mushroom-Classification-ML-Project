import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# Node class for the decision tree
class Node:
    def __init__(self, depth=0, max_depth=None):
        self.left = None   # Pointer to left child node
        self.right = None  # Pointer to right child node
        self.is_leaf = False  # Boolean indicating whether the node is a leaf node
        self.prediction = None  # Stores the class prediction if the node is a leaf
        self.depth = depth  # Current depth of the node in the tree
        self.max_depth = max_depth  # Maximum allowed depth for the tree
        self.feature_index = None  # Index of the feature used for splitting at this node
        self.threshold = None  # Threshold for numerical features
        self.categories = None  # Categories for categorical features
        self.is_categorical = False  # Boolean indicating if the split is on a categorical feature
        self.impurity = None  # Impurity measure (e.g., Gini, Entropy) at the node
        self.n_samples = None  # Number of samples reaching the node
        self.impurity_decrease = 0.0  # Impurity decrease at the node

# DecisionTree class with support for categorical features
class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0, max_features=None):
        self.root = None
        self.criterion = criterion  # Function to measure the quality of a split
        self.max_depth = max_depth  # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split an internal node
        self.min_impurity_decrease = min_impurity_decrease  # Minimum impurity decrease required for splitting
        self.n_classes = None
        self.feature_types = None  # To store feature types (categorical or numerical)
        self.n_features = None
        self.max_features = max_features

    def fit(self, X, y, feature_types):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.feature_types = feature_types
        self.root = self._grow_tree(X, y)
        self._assign_n_samples_and_impurity(self.root, X, y)  # Assign n_samples and impurity for pruning

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        node = Node(depth=depth, max_depth=self.max_depth)

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (num_samples < self.min_samples_split) or \
           (np.unique(y).size == 1):
            node.is_leaf = True
            node.prediction = self._most_common_label(y)
            return node

        # Find the best split
        best_feat, best_thresh, best_categories, best_gain = self._best_split(X, y)

        # If no valid split was found
        if best_gain is None or best_gain < self.min_impurity_decrease:
            node.is_leaf = True
            node.prediction = self._most_common_label(y)
            return node

        # Store impurity decrease at this node
        node.impurity_decrease = best_gain

        # Split the tree recursively
        if self.feature_types[best_feat] == 'categorical':
            indices_left = np.isin(X[:, best_feat], best_categories)
        else:
            indices_left = X[:, best_feat] <= best_thresh
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        node.feature_index = best_feat
        node.is_categorical = self.feature_types[best_feat] == 'categorical'
        if node.is_categorical:
            node.categories = best_categories
        else:
            node.threshold = best_thresh
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feat = None
        best_thresh = None
        best_categories = None

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Precompute impurity of the parent node
        parent_impurity = self._impurity(y)
        if self.max_features == 'sqrt':
            n_f = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_f = max(1, int(np.log2(n_features)))
        else:
            n_f = n_features

        feature_indices = np.random.choice(n_features, n_f, replace=False)
        for feature_index in feature_indices:
            feature_type = self.feature_types[feature_index]
            X_column = X[:, feature_index]

            if feature_type == 'numerical':
                # Numerical feature
                sort_idx = np.argsort(X_column)
                X_column_sorted = X_column[sort_idx]
                y_sorted = y[sort_idx]

                # Potential split positions where feature value changes
                unique_values = np.unique(X_column_sorted)
                if len(unique_values) == 1:
                    continue  # Cannot split further

                # Compute cumulative counts for each class
                class_counts = np.zeros((len(y_sorted), n_classes))
                for c in range(n_classes):
                    y_binary = (y_sorted == c).astype(int)
                    class_counts[:, c] = np.cumsum(y_binary)

                total_counts = class_counts[-1, :]

                for i in range(1, n_samples):
                    if X_column_sorted[i] == X_column_sorted[i - 1]:
                        continue  # Skip identical feature values

                    left_counts = class_counts[i - 1, :]
                    right_counts = total_counts - left_counts

                    # Avoid empty splits
                    if np.sum(left_counts) == 0 or np.sum(right_counts) == 0:
                        continue

                    left_impurity = self._impurity_from_counts(left_counts)
                    right_impurity = self._impurity_from_counts(right_counts)

                    n_left = i
                    n_right = n_samples - i
                    child_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity

                    ig = parent_impurity - child_impurity

                    if ig > best_gain:
                        best_gain = ig
                        best_feat = feature_index
                        best_thresh = (X_column_sorted[i] + X_column_sorted[i - 1]) / 2
                        best_categories = None

            else:
                # Categorical feature
                categories, counts = np.unique(X_column, return_counts=True)
                if len(categories) == 1:
                    continue  # Cannot split further

                # Compute mean target value for each category
                category_means = {}
                for category in categories:
                    y_category = y[X_column == category]
                    category_means[category] = np.mean(y_category)

                # Sort categories based on mean target value
                sorted_categories = sorted(categories, key=lambda c: category_means[c])

                for i in range(1, len(sorted_categories)):
                    left_categories = sorted_categories[:i]

                    left_indices = np.isin(X_column, left_categories)
                    right_indices = ~left_indices

                    y_left = y[left_indices]
                    y_right = y[right_indices]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue

                    left_impurity = self._impurity(y_left)
                    right_impurity = self._impurity(y_right)

                    n_left = len(y_left)
                    n_right = len(y_right)
                    child_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity

                    ig = parent_impurity - child_impurity

                    if ig > best_gain:
                        best_gain = ig
                        best_feat = feature_index
                        best_thresh = None
                        best_categories = left_categories

        if best_gain == -np.inf:
            return None, None, None, None

        return best_feat, best_thresh, best_categories, best_gain

    def _impurity_from_counts(self, counts):
        total = np.sum(counts)
        if total == 0:
            return 0
        if len(counts) < 2:
            counts = np.append(counts, 0)
        p = counts[1] / total  # Assuming binary classification with classes 0 and 1
        if self.criterion == 'gini':
            return 2 * p * (1 - p)
        elif self.criterion == 'entropy':
            return - (p / 2) * np.log2(p + 1e-9) - ((1 - p) / 2) * np.log2(1 - p + 1e-9)
        elif self.criterion == 'squared_impurity':
            return np.sqrt(p * (1 - p))
        else:
            raise ValueError("Unknown criterion.")

    def _impurity(self, y):
        total = len(y)
        if total == 0:
            return 0
        counts = np.bincount(y)
        if len(counts) < 2:
            counts = np.append(counts, 0)
        p = counts[1] / total  # Assuming binary classification with classes 0 and 1
        if self.criterion == 'gini':
            return 2 * p * (1 - p)
        elif self.criterion == 'entropy':
            return - (p / 2) * np.log2(p + 1e-9) - ((1 - p) / 2) * np.log2(1 - p + 1e-9)
        elif self.criterion == 'squared_impurity':
            return np.sqrt(p * (1 - p))
        else:
            raise ValueError("Unknown criterion.")

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node.is_leaf:
            return node.prediction
        if node.is_categorical:
            if inputs[node.feature_index] in node.categories:
                return self._predict(inputs, node.left)
            else:
                return self._predict(inputs, node.right)
        else:
            if inputs[node.feature_index] <= node.threshold:
                return self._predict(inputs, node.left)
            else:
                return self._predict(inputs, node.right)

    # Assign number of samples and impurity to nodes for pruning
    def _assign_n_samples_and_impurity(self, node, X, y):
        node.n_samples = len(y)
        node.impurity = self._impurity(y)

        if node.is_leaf:
            return

        if node.is_categorical:
            indices_left = np.isin(X[:, node.feature_index], node.categories)
        else:
            indices_left = X[:, node.feature_index] <= node.threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        self._assign_n_samples_and_impurity(node.left, X_left, y_left)
        self._assign_n_samples_and_impurity(node.right, X_right, y_right)

    # Pruning method
    def prune(self, X_val, y_val):
        self._prune_tree(self.root, X_val, y_val)

    def _prune_tree(self, node, X_val, y_val):
        if node.is_leaf:
            return

        # Prune left and right subtrees first
        self._prune_tree(node.left, X_val, y_val)
        self._prune_tree(node.right, X_val, y_val)

        # If both children are leaves, consider pruning
        if node.left.is_leaf and node.right.is_leaf:
            # Get validation predictions before pruning
            y_pred_before = self.predict(X_val)
            error_before = np.mean(y_pred_before != y_val)

            # Save the children
            left_child = node.left
            right_child = node.right

            # Prune the node
            node.is_leaf = True
            node.left = None
            node.right = None
            node.prediction = self._most_common_label(y_val)

            # Get validation predictions after pruning
            y_pred_after = self.predict(X_val)
            error_after = np.mean(y_pred_after != y_val)

            # Decide whether to keep pruning
            if error_after > error_before:
                # Revert pruning
                node.is_leaf = False
                node.left = left_child
                node.right = right_child
                node.prediction = None

    # Compute feature importances
    def compute_feature_importances(self):
        importances = np.zeros(self.n_features)

        def traverse(node):
            if node.is_leaf:
                return

            # Accumulate the impurity decrease for the feature used at this node
            importances[node.feature_index] += node.impurity_decrease

            traverse(node.left)
            traverse(node.right)

        traverse(self.root)

        # Normalize the importances
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances /= total_importance

        return importances