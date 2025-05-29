"""
Configuration file for machine learning models, including classifiers and their parameter grids.
This file centralizes all model configurations for better maintainability and reusability.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier

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
PARAM_GRIDS = {
    "MLP": {
        "classifier__hidden_layer_sizes": [(100,), (100, 50)],
        "classifier__alpha": [1e-4, 1e-3, 1e-2],
    },
    
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": ["scale", "auto", 0.01]
    },
    
    "RandomForest": {
        # Millors paràmetres trobats
        "classifier__n_estimators": [1163],
        "classifier__max_depth": [8],
        "classifier__max_features": ["log2"],
        "classifier__min_samples_leaf": [3],
        "classifier__min_samples_split": [5],
        "classifier__class_weight": ["balanced"]
    },
    
    "BalancedRandomForest": {
        "classifier__n_estimators": [1163],
        "classifier__max_depth": [8],
        "classifier__max_features": ["log2"],
        "classifier__min_samples_leaf": [3],
        "classifier__min_samples_split": [5],
        "classifier__class_weight": ["balanced"]
    },
    
    "GradientBoosting": {
        "classifier__n_estimators": [200, 400],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]
    },
    
    "LGBM": {
        # Millors paràmetres trobats
        'classifier__colsample_bytree': [0.9116586907214196], 
        'classifier__learning_rate': [0.09826363431893397], 
        'classifier__min_child_samples': [11], 
        'classifier__n_estimators': [508], 
        'classifier__num_leaves': [49], 
        'classifier__reg_alpha': [0.3629778394351197], 
        'classifier__reg_lambda': [0.8971102599525771], 
        'classifier__subsample': [0.9774172848530235]
    }
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
