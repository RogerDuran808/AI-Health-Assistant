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

from ai_health_assistant.utils.train_helpers import train_models, append_results, mat_confusio, plot_learning_curve

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
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False)
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

    # Els millors parametres trobats els posare al costat per tenir una referencia. Best F1 = 0.5598, Acc= 0.5022 
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
    }
}

results = []
models = {}

# Entrenament del model 
model_name = "BalancedRandomForest" # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest
clf = CLASSIFIERS[model_name]


pipeline = ImbPipeline([
    ("smote", BorderlineSMOTE(random_state=42, sampling_strategy=0.95)),
    ("classifier", clf)
])

# Farem proves també amb el BalancedRandomForestClassifier
pipeline_no_smote = ImbPipeline([
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
    search_type='grid' # 'grid' quan fem search amb parametres especifics, sino predefinit 'random' que fa un randomsearch
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
    y
)

mat_confusio(
    model_name,
    y_test,
    y_test_pred
)

print(f"\n\nMillors parametrs pel model - {model_name}:\n")
print(results_df[results_df['Model'] == model_name]['Best Params'].values[0])
print('\n')

# Guradem el model a la carpeta models
joblib.dump(best_est, f'models/{model_name}_TIRED.joblib')


