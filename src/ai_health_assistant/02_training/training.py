import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel

from ai_health_assistant.utils.train_helpers import train_models, append_results, mat_confusio, plot_learning_curve, save_model, update_metrics_file
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.prep_helpers import TARGET, build_preprocessor, FEATURES

import warnings
warnings.filterwarnings('ignore')

#=========================================================

# Definicions d'entrenament
model_name = "BalancedRandomForest" # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest, LGBM
balance_name = 'SMOTETomek' # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE, SMOTE
pipeline_name = 'no_balance' # basic, no_balance
features = 'all' # all, top10_perm, top10_fi

#===========================================================

# Llegim les dades amb fe aplicat
df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')

# Seleccio de features, posibles columnes trobades a 02_LifeSnaps_Training_Experiments.ipynb o altres proves
top15_perm = ['bmi', 'recovery_factor', 'minutesAsleep', 'full_sleep_breathing_rate', 'daily_temperature_variation', 'minutes_in_default_zone_1', 'wake_after_sleep_pct', 'calories', 'active_to_rest_transition', 'rmssd', 'sleep_activity_balance', 'deep_sleep_score', 'steps_norm_cal', 'sleep_wake_ratio', 'sleep_rem_ratio']
top10_fi = ['calories', 'bmi_hr_interaction', 'bmi', 'resting_hr', 'steps_norm_cal', 'daily_temperature_variation', 'recovery_factor', 'hr_zone_variability', 'lightly_active_minutes', 'minutesAsleep']    

# ---------------------------------------------------

# Seleccio de features
if features == 'top15_perm':
    FEATURES = top15_perm
elif features == 'top10_fi':
    FEATURES = top10_fi
else:
    FEATURES = FEATURES

# --------------------------------------------------------

# Definició del train / test    
X_train = df_train[FEATURES]
y_train = df_train[TARGET]
    
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# ----------------------------------------------

# Construim el preprocessador
preprocessor = build_preprocessor(df_train, FEATURES)

# Definim el classifier i els parametres
clf, param_grid = get_classifier_config(model_name)

# Obtenim el metode de balanceig
balancing_method = BALANCING_METHODS[balance_name]

#----------------------------------

# Llsta dels resultats
results = []

#--------------------------------------------------

# Selecció del pipeline
if pipeline_name == 'no_balance':
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
else: # basic
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("balancing", balancing_method),
        ("classifier", clf)
    ])

#--------------------------------------------------

# Entrenament del model - Cerca de parametres segons depenent del grid o random search
best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    pipeline,
    param_grid,
    n_iter=10,
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


