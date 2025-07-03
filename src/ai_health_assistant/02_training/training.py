import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import precision_recall_curve, f1_score

from ai_health_assistant.utils.train_helpers import train_models, append_results, mat_confusio, plot_learning_curve, save_model, update_metrics_file
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.prep_helpers import TARGET, build_preprocessor, FEATURES

import warnings
warnings.filterwarnings('ignore')

#=========================================================

# Definicions d'entrenament
model_name = "LGBM"  # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest, LGBM
balance_name = 'SMOTETomek'  # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE, SMOTE
pipeline_name = 'balance'  # balance, no_balance
features = 'top13_perm'  # all, top15_perm, top13_perm, top10_fi, sel_manual

#=========================================================

# Llegim les dades amb feature engineering aplicat
df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')

# Selecció de característiques, possibles columnes trobades a 02_LifeSnaps_Training_Experiments.ipynb o altres proves
top15_perm = ['bmi', 'recovery_factor', 'minutesAsleep', 'full_sleep_breathing_rate', 'daily_temperature_variation', 'minutes_in_default_zone_1', 'wake_after_sleep_pct', 'calories', 'active_to_rest_transition', 'rmssd', 'sleep_activity_balance', 'deep_sleep_score', 'steps_norm_cal', 'sleep_wake_ratio', 'sleep_rem_ratio']
top10_fi = ['calories', 'bmi_hr_interaction', 'bmi', 'resting_hr', 'steps_norm_cal', 'daily_temperature_variation', 'recovery_factor', 'hr_zone_variability', 'lightly_active_minutes', 'minutesAsleep']
top13_perm = ['bmi', 'recovery_factor', 'minutesAsleep', 'full_sleep_breathing_rate', 'daily_temperature_variation', 'minutes_in_default_zone_1', 'wake_after_sleep_pct', 'calories', 'active_to_rest_transition', 'rmssd', 'sleep_activity_balance', 'deep_sleep_score', 'steps_norm_cal']
top_perm_2 = ['num__bmi', 'num__daily_temperature_variation', 'num__resting_hr', 'num__sleep_efficiency', 'num__rmssd', 'num__sleep_deep_ratio', 'cat__gender_FEMALE', 'num__lightly_active_minutes', 'num__calories', 'num__bmi_hr_interaction']
sel_manual = ['bmi', 'calories', 'resting_hr', 'rmssd', 'sleep_efficiency', 'minutesAsleep', 'minutesAwake']

# ---------------------------------------------------

# Selecció de característiques segons la variable 'features'
if features == 'top15_perm':
    FEATURES = top15_perm
elif features == 'top10_fi':
    FEATURES = top10_fi
elif features == 'top13_perm':
    FEATURES = top13_perm
elif features == 'sel_manual':
    FEATURES = sel_manual
else:
    FEATURES = FEATURES

# --------------------------------------------------------

# Definició de train/test
X_train = df_train[FEATURES]
y_train = df_train[TARGET]
    
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# ----------------------------------------------

# Construïm el preprocesador
preprocessor = build_preprocessor(df_train, FEATURES)

# Definim el classificadors i els paràmetres
clf, param_grid = get_classifier_config(model_name)

# Obtenim el mètode de balanç
balancing_method = BALANCING_METHODS[balance_name]

#----------------------------------

# Llista de resultats
results = []

#--------------------------------------------------

# Selecció del pipeline
if pipeline_name == 'no_balance':
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
else:  # balance
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("balancing", balancing_method),
        ("classifier", clf)
    ])

#--------------------------------------------------

# Entrenament del model – cerca de paràmetres segons el tipus de cerca (grid o random)
best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    pipeline,
    param_grid,
    n_iter=25,
    search_type='grid',  # 'grid' per a cerca amb paràmetres específics; sinó, randomsearch per defecte
)

# Guardem els resultats en un DataFrame
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

# Guardem la corba d'aprenentatge si save='yes'
plot_learning_curve(
    model_name,
    best_est,
    X_train,
    y_train,
    save='yes'
)

# Guardem les mètriques del model amb els millors paràmetres a 03_training/metrics.csv
update_metrics_file(results_df)

# Amb la funció definida, guardem el model entrenat a la carpeta de models local
# i a la carpeta de models de la nostra webapp, per poder-lo carregar si save_external='yes'
save_model(best_est, model_name, save_external='no')
