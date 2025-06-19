"""
Fitxer de configuració dels models d'aprenentatge automàtic, incloent classificadors i els seus param grids.
Aquest fitxer centralitza totes les configuracions dels models per millorar la maintainabilitat i la reutilització.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier

from scipy.stats import randint, uniform, loguniform
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

##################### CLASSIFIERS #####################
CLASSIFIERS = {
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "SVM": SVC(random_state=42, probability=True, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False),
    "LGBM": LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, importance_type='gain', verbose=0, objective='binary')
}

##################### PARAM GRIDS #####################
# Param grids o distribucions per fer el GridSearch o RandomSearch
# Un cop s'han trobat els millors parametres, s'han guardat per no fer constantment la busqueda
# Anirem comprovant i ajustant els parametres dels models per veures quins en donen millors resultats.
# Per això anirem ajustant cada classifier avaluat.
PARAM_GRIDS = {
    "MLP": {
        "classifier__activation": ["tanh"],
        "classifier__alpha": [0.0006621289195469929],
        "classifier__batch_size": [256],
        "classifier__early_stopping": [True],
        "classifier__hidden_layer_sizes": [(70, 30)],
        "classifier__learning_rate_init": [0.0033388912653937236],
        "classifier__max_iter": [200],
        "classifier__solver": ["adam"],

        # # Parametres pel RandomSearch - MLP:
        # "classifier__hidden_layer_sizes": [(50,), (100,), (70, 30)],
        # "classifier__activation": ["relu", "tanh"],
        # "classifier__alpha": loguniform(1e-5, 1e-3),
        # "classifier__learning_rate_init": loguniform(1e-3, 1e-2),
        # "classifier__batch_size": [128, 256],
        # "classifier__max_iter": [200],      
        # "classifier__early_stopping": [True],
        # "classifier__solver": ["adam"],   
    },
    
    "SVM": {
        # Millors paràmetres trobats
        "classifier__C": [10],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.01]

        # # Paràmetres pel GridSearch - SVM:
        # "classifier__C": [0.1, 1, 10],
        # "classifier__kernel": ["rbf"],
        # "classifier__gamma": ["scale", "auto", 0.01]

    },
    
    "RandomForest": {
        # Millors paràmetres trobats
        "classifier__max_depth": [6],
        "classifier__max_features": ["log2"],
        "classifier__min_samples_leaf": [2],
        "classifier__n_estimators": [517],
        "classifier__class_weight": ["balanced_subsample"]

        # Parametres pel RandomSearch - RandomForest:
        # "classifier__n_estimators": randint(50, 151),  
        # "classifier__max_depth": randint(4, 11), 
        # "classifier__min_samples_split": randint(2, 11),
        # "classifier__min_samples_leaf": randint(1, 6),
        # "classifier__max_features": ["sqrt", "log2"],
        # "classifier__bootstrap": [True],
        # "classifier__criterion": ["gini", "entropy"],

        
    },
    
    "BalancedRandomForest": {

        # # # Millors paràmetres trobats v1 per trobar el TOP13PERM que dona un F1 de 60.37% amb LGBM
        # "classifier__class_weight": ["balanced"],
        # "classifier__max_depth": [8],
        # "classifier__max_features": ["log2"],
        # "classifier__min_samples_leaf": [3],
        # "classifier__min_samples_split": [5],
        # "classifier__n_estimators": [1163]

        # # Millors paràmetres trobats v2
        "classifier__class_weight": ["balanced_subsample"],
        "classifier__max_depth": [10],
        "classifier__max_features": ["sqrt"],
        "classifier__min_samples_leaf": [11],
        "classifier__min_samples_split": [4],
        "classifier__n_estimators": [1021]

        # # Parametres pel RandomSearch - BalancedRandomForest:
        # "classifier__n_estimators": randint(500, 1200),
        # "classifier__max_depth": randint(5, 16),
        # "classifier__max_features": ["sqrt", "log2", 0.5],
        # "classifier__min_samples_leaf": randint(5, 10),
        # "classifier__min_samples_split": randint(3,6),
        # "classifier__class_weight": ["balanced", "balanced_subsample"],

        

    },
    
    "GradientBoosting": {
        # Millors paràmetres trobats:
        'classifier__learning_rate': [0.07720558711161865],
        'classifier__loss': ['log_loss'],
        'classifier__max_depth': [2],
        'classifier__max_features': ['sqrt'],
        'classifier__min_samples_leaf': [5],
        'classifier__min_samples_split': [8],
        'classifier__n_estimators': [66],
        'classifier__subsample': [0.842289601399309]

        # Parametres pel RandomSearch - GradientBoosting:
        # "classifier__n_estimators": randint(50, 201), 
        # "classifier__learning_rate": loguniform(0.05, 0.3),
        # "classifier__max_depth": randint(2, 4), 
        # "classifier__min_samples_split": randint(2, 11),
        # "classifier__min_samples_leaf": randint(1, 6),
        # "classifier__subsample": uniform(0.8, 0.2), 
        # "classifier__max_features": ["sqrt", "log2"],
        # "classifier__loss": ["log_loss"],
    },
    
    "LGBM": {
        # # Paràmetres trobats per LGBM v1:
        # 'classifier__colsample_bytree': [0.2678280051592841], 
        # 'classifier__learning_rate': [0.015273394899450402], 
        # 'classifier__min_child_samples': [5], 
        # 'classifier__n_estimators': [673], 
        # 'classifier__num_leaves': [94], 
        # 'classifier__reg_alpha': [0.24018504093317733], 
        # 'classifier__reg_lambda': [1.4779290759982566], 
        # 'classifier__subsample': [0.7651084549069112],
        # "classifier__boosting_type": ["dart"],
        # "classifier__scale_pos_weight": [1.6] # Aprox 62/38 classe desbalancejada

        # # # Busqueda de paràmetres RandomSearch - LGBM:
        # "classifier__boosting_type": ["dart", "gbdt"],
        # "classifier__n_estimators":   randint(400, 1200),
        # "classifier__learning_rate":  loguniform(5e-3, 4e-2),
        # "classifier__num_leaves":     randint(60, 160),
        # "classifier__max_depth":      [-1, 5, 7, 9],
        # "classifier__min_child_samples": randint(5, 40),
        # "classifier__subsample":      uniform(0.6, 0.95),       # 0.60-0.95
        # "classifier__colsample_bytree": uniform(0.2, 0.7),      # 0.20-0.70
        # "classifier__reg_alpha":      loguniform(1e-3, 1.0),
        # "classifier__reg_lambda":     loguniform(1e-3, 5.0),
        # "classifier__scale_pos_weight": uniform(1.2, 2.5),      # 1.2-2.5
        # "classifier__min_split_gain": uniform(0.0, 0.3),     

        # Paràmetres trobats per LGBM v2 (amb topperm13 un f1 de 60.37%):
        'classifier__colsample_bytree': [0.53],
        'classifier__learning_rate': [0.0058],
        'classifier__max_depth': [9],
        'classifier__min_child_samples': [34],
        'classifier__min_split_gain': [0.06],
        'classifier__n_estimators': [591],
        'classifier__num_leaves': [119],
        'classifier__reg_alpha': [0.0036],
        'classifier__reg_lambda': [0.0134],
        'classifier__scale_pos_weight': [1.62],
        'classifier__subsample': [0.75],
        
    }
}

##################### BALANCING METHODS #####################
# Definim varis metodes de balanceig per comparar
BALANCING_METHODS = {
    "SMOTETomek": SMOTETomek(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42, k_neighbors=5),
    "SMOTE": SMOTE(random_state=42)
}

################### FUNCIO OBTENCIO DEL CLF I PARAMS ###################
def get_classifier_config(model_name):
    """
    Obtenim el classifier i els seus parametres pel seu nom\n
        
    Return:
    - classifier: classifier seleccionat
    - param_grid: parametres del classifier seleccionat
    """
    if model_name not in CLASSIFIERS:
        raise ValueError(f"El model no està en la llista: {model_name}. Models disponibles: {list(CLASSIFIERS.keys())}")
        
    return CLASSIFIERS[model_name], PARAM_GRIDS.get(model_name, {})
