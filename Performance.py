import pandas as pd
import numpy as np

# Functions for evaluation metrics
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

