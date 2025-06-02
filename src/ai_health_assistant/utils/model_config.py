"""
Configuration file for machine learning models, including classifiers and their parameter grids.
This file centralizes all model configurations for better maintainability and reusability.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier

from scipy.stats import randint, uniform
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE


# Base classifiers without parameters
CLASSIFIERS = {
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "SVM": SVC(random_state=42, probability=True),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False),
    "LGBM": LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, importance_type='gain', verbose=0)
}

# Param grids o distribucions per fer el GridSearch o RandomSearch
# Un cop s'han trobat els millors parametres, s'han guardat per no fer constantment la busqueda
# Anirem comprovant i ajustant els parametres dels models per veures quins en donen millors resultats.
# Per això anirem ajustant cada classifier avaluat.

PARAM_GRIDS = {
    "MLP": {
        "classifier__hidden_layer_sizes": [(100,), (100, 50)],
        "classifier__alpha": [1e-4, 1e-3, 1e-2],

        # # Parametres pel RandomSearch - MLP:
        # "classifier__hidden_layer_sizes": [(100,), (100, 50), (100, 50, 25)],
        # "classifier__alpha": uniform(1e-6, 1e-2)
    },
    
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": ["scale", "auto", 0.01]

        # # Parametres pel RandomSearch - SVM:
        # "classifier__C": uniform(0.1, 10),
        # "classifier__kernel": ["rbf", "linear", "poly", "sigmoid"],
        # "classifier__gamma": uniform(0.01, 1)
    },
    
    "RandomForest": {
        # Millors paràmetres trobats
        "classifier__n_estimators": [765],
        "classifier__max_depth": [4],
        "classifier__max_features": [0.43559419734887395],
        "classifier__min_samples_leaf": [1],
        "classifier__min_samples_split": [2],
        "classifier__class_weight": ["balanced_subsample"]

        # # Parametres pel RandomSearch - RandomForest:
        # "classifier__n_estimators": randint(400, 600),
        # "classifier__max_depth": randint(5, 7),
        # "classifier__max_features": ["sqrt", "log2", 0.5],
        # "classifier__min_samples_leaf": randint(1, 3),
        # "classifier__min_samples_split": randint(2, 8),
        # "classifier__class_weight": ["balanced", "balanced_subsample"]
    },
    
    "BalancedRandomForest": {
        "classifier__n_estimators": [1163],
        "classifier__max_depth": [8],
        "classifier__max_features": ["log2"],
        "classifier__min_samples_leaf": [3],
        "classifier__min_samples_split": [5],
        "classifier__class_weight": ["balanced"]

        # # Parametres pel RandomSearch - BalancedRandomForest:
        # "classifier__n_estimators": randint(400, 1600),
        # "classifier__max_depth": randint(5, 10),
        # "classifier__max_features": ["sqrt", "log2", 0.5],
        # "classifier__min_samples_leaf": randint(1, 5),
        # "classifier__min_samples_split": randint(2, 8),
        # "classifier__class_weight": ["balanced", "balanced_subsample"]
    },
    
    "GradientBoosting": {
        "classifier__n_estimators": [200, 400],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]

        # # Busqueda de paràmetres RandomSearch - GradientBoosting:
        # "classifier__n_estimators": randint(500, 1000),
        # "classifier__learning_rate": uniform(0.01, 0.1),
        # "classifier__max_depth": randint(3, 10)
    },
    
    "LGBM": {
        # Millors paràmetres trobats:
        'classifier__colsample_bytree': [0.2678280051592841], 
        'classifier__learning_rate': [0.015273394899450402], 
        'classifier__min_child_samples': [5], 
        'classifier__n_estimators': [673], 
        'classifier__num_leaves': [94], 
        'classifier__reg_alpha': [0.24018504093317733], 
        'classifier__reg_lambda': [1.4779290759982566], 
        'classifier__subsample': [0.7651084549069112],
        "classifier__boosting_type": ["dart"],
        "classifier__scale_pos_weight": [1.6] # Aprox 62/38 classe desbalancejada

        # # Busqueda de paràmetres RandomSearch - LGBM:
        # "classifier__n_estimators": randint(300, 1200),
        # "classifier__learning_rate": uniform(0.01, 0.2),
        # "classifier__num_leaves": randint(30, 150),
        # "classifier__reg_alpha": uniform(0, 0.5),
        # "classifier__reg_lambda": uniform(0, 1.5),
        # "classifier__min_child_samples": randint(5, 20),
        # "classifier__subsample": uniform(0.2, 1.5),
        # "classifier__colsample_bytree": uniform(0.2, 1.5),
        # "classifier__boosting_type": ["dart"],
        # "classifier__scale_pos_weight": [1.6] # Aprox 62/38 classe desbalancejada
    }
}

# Definimos varios métodos de balanceo para comparar
BALANCING_METHODS = {
    "SMOTETomek": SMOTETomek(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42, k_neighbors=5)
}

def get_classifier_config(model_name):
    """
    Obtenim el classifier i els seus parametres pel seu nom\n
        
    Returns:
        classifier, param_grid del model seleccionat
    """
    if model_name not in CLASSIFIERS:
        raise ValueError(f"Model no en la llista: {model_name}. Models disponibles: {list(CLASSIFIERS.keys())}")
        
    return CLASSIFIERS[model_name], PARAM_GRIDS.get(model_name, {})
