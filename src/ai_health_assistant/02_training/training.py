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
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.prep_helpers import FEATURES, TARGET, build_preprocessor

import warnings
warnings.filterwarnings('ignore')

#---------------------------------------------------------

# Definim el model i el balanceig
model_name = "LGBM" # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest, LGBM
balance_name = 'SMOTETomek' # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE

#---------------------------------------------------------


df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')
    
X_train = df_train[FEATURES]
y_train = df_train[TARGET]
    
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

preprocessor = build_preprocessor(X_train)

# Definim el classifier i els parametres
clf, param_grid = get_classifier_config(model_name)

results = []
models = {}


# Amb el que he obtingut millors resultats es SMOTETomek
balancing_method = BALANCING_METHODS[balance_name]

# Selecció de les millors caracteristiques 
feature_selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))

#-------------------------------------------------------------------------------------


# Obting un F1=62.57 i un acc= 72%, amb un macro av. 70.14% [BalancedRandomForest]
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
advanced_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("balancing", balancing_method),
    ("feature_selection", feature_selector),
    ("classifier", clf)
])


best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    pipeline,
    param_grid,
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

mat_confusio(
    model_name,
    y_test,
    y_test_pred,
    save='yes'
)

plot_learning_curve(
    model_name,
    best_est,
    X_train,
    y_train,
    save='yes'
)


print(f"\n\nMillors parametrs pel model - {model_name}:\n")
print(results_df[results_df['Model'] == model_name]['Best Params'].values[0])
print('\n')

# Guradem el model a la carpeta models
# Amb la funcio definida, guardem el model entrenat a la carpeta de models local
# i a la carpeta de models de la nostre webapp, per poder-lo carregar desde alla.
save_model(best_est, model_name, save_external='no')


