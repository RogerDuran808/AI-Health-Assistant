import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from scipy.stats import randint, uniform
import joblib

from sklearn.model_selection import train_test_split

from imblearn.ensemble import BalancedRandomForestClassifier


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

from sklearn.metrics import make_scorer, fbeta_score

from ai_health_assistant.utils.train_helpers import train_models, append_results, mat_confusio, plot_learning_curve, save_model

import warnings
warnings.filterwarnings('ignore')


################ Quan tinguem la bona forma de entrenar el model ##################


# Load the dataset
df = pd.read_csv('data/df_preprocessed.csv')

# Comprovem quina es les estructura de les nostres dades faltants en el target
TARGET = 'TIRED'

# Difinim X i el target y
# Prediccio de TIRED
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Estratifiquem respecte un dels targets per tal d'assegurar el bon split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Define classifiers
CLASSIFIERS = {
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "SVM": SVC(random_state=42, probability=True),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False),
    "LGBM": LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, importance_type='gain')
}

# Param grids pel GridSearchCV
# Complexitat reduida per tal que no porti un temps exegerat de execució
# Anirem comprovant i ajustant els parametres dels models per veures quins en donen millors resultats.
# Per això anirem ajustant cada possible classifier
PARAM_GRIDS = {
    "MLP": {
        "classifier__hidden_layer_sizes": [(100,), (100, 50)],
        "classifier__alpha": [1e-4, 1e-3, 1e-2],

    },
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf"], # Si hi ha bons resultats provar d'altres
        "classifier__gamma": ["scale", "auto", 0.01]
    },

    
    "RandomForest": {
        # Parametre pel GridSearch (proves dels millors parametres)
        "classifier__n_estimators": [1163], # [1163]
        "classifier__max_depth": [8], # [8]
        "classifier__max_features": ["log2"], # ["log2"]
        "classifier__min_samples_leaf": [3], # [3]
        "classifier__min_samples_split": [5], # [5]
        "classifier__class_weight": ["balanced"] # ["balanced"]

        # # Parametres pel RandomSearch
        # "classifier__n_estimators": randint(400, 600),
        # "classifier__max_depth": randint(5, 7),
        # "classifier__max_features": ["sqrt", "log2", 0.5],
        # "classifier__min_samples_leaf": randint(1, 3),
        # "classifier__min_samples_split": randint(2, 8),
        # "classifier__class_weight": ["balanced", "balanced_subsample"]

    },

    # Els millors parametres trobats els posare al costat per tenir una referencia. Best F1 = 0.5648, Acc= 0.5087 
    "BalancedRandomForest": {
        "classifier__n_estimators":      [1163], # [1163]
        "classifier__max_depth":         [8], # [8]
        "classifier__max_features":      ["log2"], # ["log2"]
        "classifier__min_samples_leaf":  [3], # [3]
        "classifier__min_samples_split": [5], # [5]
        "classifier__class_weight":      ["balanced"], # ["balanced"]
    },
    "GradientBoosting": {
        "classifier__n_estimators": [200, 400],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]
    },
    # Reduir la complexitat del GridSearchCV
    "LGBM": {
    "classifier__n_estimators": [800, 1000, 1200],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__num_leaves": [31, 63, 127],
    "classifier__reg_alpha": [0, 0.1, 0.5],
    "classifier__reg_lambda": [0, 0.1, 1.0],
    "classifier__min_child_samples": [5, 10, 20],
    "classifier__subsample": [0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.8, 0.9, 1.0]
}
}

results = []
models = {}

# Entrenament del model 
model_name = "BalancedRandomForest" # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest
clf = CLASSIFIERS[model_name]




#################################################################################
# ALTRES METODES DE BALANCEJAMENT
balancing_method = SMOTETomek(random_state=42)  # Combina oversampling y undersampling

# Implementar selección de características
feature_selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))

# Score per si volem donar més importància al recall que a la precisión
f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)  
###################################################################################


# Obting un F1=62.57 i un acc= 72%, amb un macro av. 70.14% [BalancedRandomForest]
pipeline = ImbPipeline([
    ("balancing", balancing_method),
    ("classifier", clf)
])

# Obting un F1=65.29 i un acc= 59.61%%, amb un macro av. 58.49% [BalancedRandomForest]
pipeline_no_smote = ImbPipeline([
    ("classifier", clf)
])

# Pipeline avanzado, per classificadors que no tenen balanceig incorporat
advanced_pipeline = ImbPipeline([
    ("feature_selection", feature_selector),
    ("balancing", balancing_method),
    ("classifier", clf)
])

best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    pipeline_no_smote,
    PARAM_GRIDS[model_name],
    n_iter=200,
    search_type='grid', # 'grid' quan fem search amb parametres especifics, sino predefinit 'random' que fa un randomsearch
)

models[model_name] = best_est

results_df = append_results(
    results,
    model_name,
    train_report,
    test_report,
    best_params,
    best_score
)

plot_learning_curve(
    model_name,
    models,
    X,
    y,
    save='yes'
)

mat_confusio(
    model_name,
    y_test,
    y_test_pred,
    save='yes'
)

print(f"\n\nMillors parametrs pel model - {model_name}:\n")
print(results_df[results_df['Model'] == model_name]['Best Params'].values[0])
print('\n')

# Guradem el model a la carpeta models
# Amb la funcio definida, guardem el model entrenat a la carpeta de models local
# i a la carpeta de models de la nostre webapp, per poder-lo carregar desde alla.
save_model(best_est, model_name, save_external='no')


