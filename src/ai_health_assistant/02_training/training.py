import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel

from ai_health_assistant.utils.train_helpers import train_models, append_results, mat_confusio, plot_learning_curve, save_model, update_metrics_file
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.prep_helpers import TARGET, build_preprocessor, FEATURES

import warnings
warnings.filterwarnings('ignore')

#---------------------------------------------------------

# Definim el model i el balanceig
model_name = "GradientBoosting" # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest, LGBM
balance_name = 'SMOTEENN' # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE

#---------------------------------------------------------


df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')
    
X_train = df_train[FEATURES]
y_train = df_train[TARGET]
    
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

preprocessor = build_preprocessor(df_train, FEATURES)

# Definim el classifier i els parametres
clf, param_grid = get_classifier_config(model_name)

results = []

# Amb el que he obtingut millors resultats es SMOTETomek
balancing_method = BALANCING_METHODS[balance_name]

# Selecció de les millors caracteristiques 
feature_selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))

#-------------------------------------------------------------------------------------


# Obting un F1=57,93% i un acc= 72%, amb un macro av. 70.14% [BalancedRandomForest]
pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("balancing", balancing_method),
    ("classifier", clf)
])

# Obting un F1=65.29 i un acc= 59.61%%, amb un macro av. 58.49% [BalancedRandomForest]
pipeline_no_balance = ImbPipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

# Pipeline avanzado, per classificadors que no tenen balanceig incorporat
pipeline_selection = ImbPipeline([
    ("preprocessor", preprocessor),
    ("balancing", balancing_method),
    ("feature_selection", feature_selector),
    ("classifier", clf)
])

# Entrenament del model - Cerca de parametres segons depenent del grid o random search
best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    pipeline,
    param_grid,
    n_iter=200,
    search_type='grid', # 'grid' quan fem search amb parametres especifics, sino predefinit 'random' que fa un randomsearch
)

# Guardem els resultats en un df
results_df = append_results(
    results,
    model_name,
    train_report,
    test_report,
    best_params,
    best_score
)

# Guardem la matriu de confusió si save='yes'
mat_confusio(
    model_name,
    y_test,
    y_test_pred,
    save='yes'
)

# Guardem la corva d'aprenentatge si save='yes'
plot_learning_curve(
    model_name,
    best_est,
    X_train,
    y_train,
    save='yes'
)

# Guradem les mètriques del model amb els millors parametres a 03_training/metrics.csv
update_metrics_file(results_df)

# Amb la funcio definida, guardem el model entrenat a la carpeta de models local
# i a la carpeta de models de la nostre webapp, per poder-lo carregar si save_external='yes'
save_model(best_est, model_name, save_external='no')


