import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import itertools 
from DecisionTree import DecisionTree
from RandomForest import RandomForest

# Stratified data splitting
def stratified_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    unique_classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = unique_classes.shape[0]
    train_indices = []
    val_indices = []
    for class_index in range(n_classes):
        class_member_mask = y_indices == class_index
        class_indices = np.where(class_member_mask)[0]
        np.random.shuffle(class_indices)
        split_point = int(len(class_indices) * (1 - test_size))
        train_indices.extend(class_indices[:split_point])
        val_indices.extend(class_indices[split_point:])
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

# Function to check for label leakage
def check_label_leakage(X, y):
    for i in range(X.shape[1]):
        if np.array_equal(X[:, i], y):
            print(f"Warning: Feature at index {i} is identical to the labels.")
            return True
    return False

# Function to check for perfect correlation between features and labels
def check_feature_label_correlation(X, y):
    for i in range(X.shape[1]):
        if np.issubdtype(X[:, i].dtype, np.number):
            if np.std(X[:, i]) == 0:
                continue  # Skip features with no variance
            correlation = np.corrcoef(X[:, i].astype(float), y.astype(float))[0, 1]
            if np.isnan(correlation):
                continue  # Skip features with undefined correlation
            if abs(correlation) == 1.0:
                print(f"Feature at index {i} has perfect correlation with the labels.")

# Function to perform hyperparameter tuning for Decision Tree
def hyperparameter_tuning(X_train, y_train, X_val, y_val, feature_types, criteria, stopping_params):
    # Prepare all combinations of hyperparameters
    param_combinations = list(itertools.product(criteria,
                                                stopping_params['max_depth'],
                                                stopping_params['min_samples_split'],
                                                stopping_params['min_impurity_decrease']))

    # Define the function to evaluate a single combination
    def evaluate_model(params):
        criterion, max_depth, min_samples_split, min_impurity_decrease = params
        tree = DecisionTree(criterion=criterion, max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_impurity_decrease=min_impurity_decrease)
        tree.fit(X_train, y_train, feature_types)
        y_pred = tree.predict(X_val)
        error = np.mean(y_pred != y_val)
        return (params, error, tree)

    # Use joblib's Parallel to evaluate models in parallel
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(params) for params in param_combinations)

    # Sort results to match the order of param_combinations
    results_sorted = sorted(results, key=lambda x: param_combinations.index(x[0]))

    # Print the results
    for res in results_sorted:
        params, error, _ = res
        criterion, max_depth, min_samples_split, min_impurity_decrease = params
        print(f"Criterion: {criterion}, Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, "
              f"Min Impurity Decrease: {min_impurity_decrease}, Validation Error: {error}")

    # Find the best result
    best_result = min(results, key=lambda x: x[1])
    best_score = best_result[1]
    best_tree = best_result[2]
    best_params = {'criterion': best_result[0][0],
                   'max_depth': best_result[0][1],
                   'min_samples_split': best_result[0][2],
                   'min_impurity_decrease': best_result[0][3]}

    return best_tree, best_params, best_score

# Function to perform hyperparameter tuning for Random Forest
def hyperparameter_tuning_rf(X_train, y_train, X_val, y_val, feature_types, rf_params):
    # Prepare all combinations of hyperparameters
    param_combinations = list(itertools.product(
        rf_params['n_estimators'],
        rf_params['max_features'],
        rf_params['criterion'],
        rf_params['max_depth'],
        rf_params['min_samples_split'],
        rf_params['min_impurity_decrease']
    ))

    # Define the function to evaluate a single combination
    def evaluate_model(params):
        n_estimators, max_features, criterion, max_depth, min_samples_split, min_impurity_decrease = params
        forest = RandomForest(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease
        )
        forest.fit(X_train, y_train, feature_types)
        y_pred = forest.predict(X_val)
        error = np.mean(y_pred != y_val)
        return (params, error, forest)

    # Use joblib's Parallel to evaluate models in parallel
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(params) for params in param_combinations)

    # Print the results
    for res in results:
        params, error, _ = res
        n_estimators, max_features, criterion, max_depth, min_samples_split, min_impurity_decrease = params
        print(f"n_estimators: {n_estimators}, max_features: {max_features}, criterion: {criterion}, "
              f"max_depth: {max_depth}, min_samples_split: {min_samples_split}, "
              f"min_impurity_decrease: {min_impurity_decrease}, Validation Error: {error}")

    # Find the best result
    best_result = min(results, key=lambda x: x[1])
    best_score = best_result[1]
    best_forest = best_result[2]
    best_params = {
        'n_estimators': best_result[0][0],
        'max_features': best_result[0][1],
        'criterion': best_result[0][2],
        'max_depth': best_result[0][3],
        'min_samples_split': best_result[0][4],
        'min_impurity_decrease': best_result[0][5],
    }

    return best_forest, best_params, best_score

# Function to compute feature importances for Random Forest
def compute_feature_importances(forest):
    importances = np.zeros(forest.n_features)

    for tree in forest.trees:
        tree_importances = tree.compute_feature_importances()
        importances += tree_importances

    # Average the importances over all trees
    total_importance = np.sum(importances)
    if total_importance > 0:
        importances /= total_importance

    return importances