import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV

from scipy.stats import randint, uniform
from ai_health_assistant.utils.train_helpers import mat_confusio, train_models, optimize_threshold_v1
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.prep_helpers import build_preprocessor, TARGET, FEATURES


# ---------------------------------------------------------

# Definim el model a utilitzar
model_name = "LGBM" # Possibles models:"MLP", "SVM", "RandomForest", "GradientBoosting", "BalancedRandomForest", "LGBM"
balance_name = 'SMOTETomek' # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE
pipeline_name = 'basic' # basic, no_balance
features = 'all' # all, top10_perm, top10_fi

# ---------------------------------------------------------

# Load el dataset
df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')

# Seleccio de features, posibles columnes trobades a 02_LifeSnaps_Training_Experiments.ipynb o altres proves
top15_perm = ['bmi', 'recovery_factor', 'minutesAsleep', 'full_sleep_breathing_rate', 'daily_temperature_variation', 'minutes_in_default_zone_1', 'wake_after_sleep_pct', 'calories', 'active_to_rest_transition', 'rmssd', 'sleep_activity_balance', 'deep_sleep_score', 'steps_norm_cal', 'sleep_wake_ratio', 'sleep_rem_ratio']
top10_fi = ['calories', 'bmi_hr_interaction', 'bmi', 'resting_hr', 'steps_norm_cal', 'daily_temperature_variation', 'recovery_factor', 'hr_zone_variability', 'lightly_active_minutes', 'minutesAsleep']    

# -------------------------------------------------------


# Seleccio de features
if features == 'top15_perm':
    FEATURES = top15_perm
elif features == 'top10_fi':
    FEATURES = top10_fi
else:
    FEATURES = FEATURES

# --------------------------------------------------------

# Prediccio de TIRED
X_temp = df_train[FEATURES]
y_temp = df_train[TARGET]

X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# -------------------------------------------------------

# Obtenim el classificador i els seus paràmetres des de la configuració centralitzada
clf, param_grid = get_classifier_config(model_name)

# Definim el balancing method
balancing_method = BALANCING_METHODS[balance_name]  # Combina oversampling i undersampling

#---------------------------------------------------------


################# 2n SPLIT pel calcul del umbral #################

# Fem una altre divisió del 80% per entrenament i per fer la validació
X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42)

preprocessor = build_preprocessor(X_train, FEATURES)

# ----------------------------------------------------------

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

# ----------------------------------------------------------

# Entrenament del model
best_est, y_train_pred, train_report, y_val_pred, val_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_val, 
    y_val,
    pipeline,
    param_grid,
    n_iter=200,
    search_type='grid'
)

#---------------------------------------------------------

#################### OPTIMITZEM UMBRAL ####################

threshold = optimize_threshold_v1(best_est, X_val, y_val, target_recall=0.65)
y_pred_optimized = (best_est.predict_proba(X_test)[:, 1] >= threshold).astype(int)

print("\n== Classification report en TEST ==")
print(classification_report(y_test, y_pred_optimized, digits=4))

# Plot de la matriu de confusió
mat_confusio(
    f"UAjust_{model_name}_v1",  
    y_test,
    y_pred_optimized,
    save='yes'
    )