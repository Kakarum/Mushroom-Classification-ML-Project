# Optimized Decision Tree and Random Forest Classifiers for Mushroom Toxicity Detection

### Machine Learning, Data Science for Economics, Unimi

### Contributors:
- [Luigi Gallo](https://github.com/Kakarum)

### Goal:
This project focuses on the classification of mushrooms as edible or poisonous using custom-built Decision Trees and Random Forests models. A comprehensive dataset describing the physical characteristics of mushrooms is analyzed, cleaned, and prepared for training. Custom implementations of tree-based models are developed from scratch, employing splitting and stopping criteria to optimize tree growth and prevent overfitting.

The hyperparameter are tuned via grid search to identify the optimal configurations and the models are subjected to pruning to refine their structures and enhance generalization. The Random Forest model is further optimized by tuning the number of estimators, maximum features considered per split, and impurity thresholds.

The performances are tested across training, validation, and test sets using metrics like accuracy, precision, recall, F1 score, and confusion matrices. The Random Forest model achieved exceptional results, with a test accuracy of 100% and flawless recall and precision, demonstrating its efficacy for this binary classification task. Then, a detailed feature importance analysis highlights the physical traits most indicative of mushroom toxicity.

### Repository Structure:
- **main.ipynb**: Main notebook that runs the project using the modules in the repository.
- **DecisionTree.py**: Contains the implementation of the Decision Tree algorithm from scratch.
- **RandomForest.py**: Module for the Random Forest classifier, including bootstrapping and aggregation logic.
- **Tuning.py**: Hyperparameter tuning utilities for tree-based models.
- **Performance.py**: Functions for evaluating model performance, including accuracy, F1 score, and confusion matrix.
- **MushroomDataset/**: Folder containing the dataset and related data handling scripts.
- **__init__.py**: Initializes the package structure for Python module imports.