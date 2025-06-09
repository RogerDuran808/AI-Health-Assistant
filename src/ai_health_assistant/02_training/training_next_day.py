'''
Entrenament de TIRED del dia després
'''

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

#=========================================================

# Definicions d'entrenament
model_name = "RandomForest"  # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest, LGBM
balance_name = 'SMOTETomek'  # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE, SMOTE
pipeline_name = 'basic'  # basic, no_balance
features = 'all'  # all, top15_perm, top10_fi

#===========================================================

# Llegim les dades amb FE aplicat
df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')

# Seleccio de features, posibles columnes trobades a 02_LifeSnaps_Training_Experiments.ipynb o altres proves
top15_perm = ['bmi', 'recovery_factor', 'minutesAsleep', 'full_sleep_breathing_rate', 'daily_temperature_variation', 
              'minutes_in_default_zone_1', 'wake_after_sleep_pct', 'calories', 'active_to_rest_transition', 'rmssd', 
              'sleep_activity_balance', 'deep_sleep_score', 'steps_norm_cal', 'sleep_wake_ratio', 'sleep_rem_ratio']
top10_fi = ['calories', 'bmi_hr_interaction', 'bmi', 'resting_hr', 'steps_norm_cal', 'daily_temperature_variation', 
            'recovery_factor', 'hr_zone_variability', 'lightly_active_minutes', 'minutesAsleep']    

# ---------------------------------------------------
# Seleccio de features
if features == 'top15_perm':
    FEATURES = top15_perm
elif features == 'top10_fi':
    FEATURES = top10_fi
else:
    FEATURES = FEATURES

# --------------------------------------------------------
# Preparem les dades per predir el dia següent
# Ordenem per data per assegurar l'ordre correcte
df_train = df_train.sort_index()
df_test = df_test.sort_index()

# Desplacem la variable objectiu un dia cap endavant
df_train[TARGET] = df_train[TARGET].shift(-1)
df_test[TARGET] = df_test[TARGET].shift(-1)

# Eliminem l'última fila de cada conjunt ja que no tindrem valor de TIRED per al dia següent
df_train = df_train.iloc[:-1]
df_test = df_test.iloc[:-1]

# Definició del train / test    
# Ens assegurem de mantenir els DataFrames de pandas per preservar els noms de les característiques
X_train = df_train[FEATURES].copy()
y_train = df_train[TARGET].copy()
    
X_test = df_test[FEATURES].copy()
y_test = df_test[TARGET].copy()

# ----------------------------------------------
# Construim el preprocessador
# Creem el preprocessador assegurant-nos que manté els noms de les característiques
preprocessor = build_preprocessor(df_train, FEATURES)


# Definim el classifier i els parametres
clf, param_grid = get_classifier_config(model_name)

# Obtenim el metode de balanceig
balancing_method = BALANCING_METHODS[balance_name]

#----------------------------------
# Llista dels resultats
results = []

#--------------------------------------------------
# Selecció del pipeline
if pipeline_name == 'no_balance':
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
else:  # basic
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("balancing", balancing_method),
        ("classifier", clf)
    ])

#--------------------------------------------------
# Entrenament del model - Cerca de paràmetres segons depenent del grid o random search
best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    pipeline,
    param_grid,
    n_iter=10,
    search_type='random',  # 'grid' quan fem search amb paràmetres específics, sinó predefinit 'random' que fa un randomsearch
)

# Guardem els resultats en un df
results_df = append_results(
    results,
    f"{model_name}_next_day",
    train_report,
    test_report,
    best_params,
    best_score
)

# Guardem la matriu de confusió si save='yes'
mat_confusio(
    f"{model_name}_next_day",
    y_test,
    y_test_pred,
    save='yes'
)

# Guardem la corva d'aprenentatge si save='yes'
plot_learning_curve(
    f"{model_name}_next_day",
    best_est,
    X_train,
    y_train,
    save='yes'
)

# Actualitzem el fitxer de mètriques amb els resultats
update_metrics_file(results_df)

# Guardem el model entrenat
save_model(best_est, f"{model_name}_next_day", save_external='no')