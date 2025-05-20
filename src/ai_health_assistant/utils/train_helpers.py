import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def grid_search(X_train, y_train, model, param_grid, scoring='f1', cv=5):
    """
    Perform grid search with cross-validation to find the best hyperparameters for a given model.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - model: The machine learning model to be tuned
    - param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    - scoring: Scoring method to evaluate the performance of the model
    - cv: Number of folds in cross-validation
    
    Returns:
    - best_model: The best model found during the grid search
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    
    return grid_search.best_estimator_