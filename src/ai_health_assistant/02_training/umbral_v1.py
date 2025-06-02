import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV

from scipy.stats import randint, uniform
from ai_health_assistant.utils.train_helpers import mat_confusio, train_models, optimize_threshold
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.prep_helpers import build_preprocessor

# Load el dataset
df_train = pd.read_csv('data/df_engineered_train.csv')
df_test = pd.read_csv('data/df_engineered_test.csv')

# ---------------------------------------------------------
# Definim el model a utilitzar
model_name = "BalancedRandomForest" # Possibles models:"MLP", "SVM", "RandomForest", "GradientBoosting", "BalancedRandomForest", "LGBM"
balance_name = 'SMOTETomek' # SMOTETomek, SMOTEENN, ADASYN, BorderlineSMOTE

# Comprovem quina es les estructura de les nostres dades faltants en el target
TARGET = 'TIRED'

# Difinim X i el target y
# Prediccio de TIRED
X_temp = df_train.drop(columns=[TARGET])
y_temp = df_train[TARGET]

X_test = df_test.drop(columns=[TARGET])
y_test = df_test[TARGET]



# Obtenim el classificador i els seus paràmetres des de la configuració centralitzada
clf, param_grid = get_classifier_config(model_name)

# Definim el balancing method
balancing_method = BALANCING_METHODS[balance_name]  # Combina oversampling i undersampling



#---------------------------------------------------------
# 2n SPLIT pel calcul del umbral
#---------------------------------------------------------

# Fem una altre divisió del 80% per entrenament i per fer la validació
X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42)

preprocessor = build_preprocessor(X_train)

# Provem els pipeline  que ens dongui millors resultats:
pipeline_1 = ImbPipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

pipeline_2 = ImbPipeline([
    ("preprocessor", preprocessor),
    ("balancing", balancing_method),
    ("classifier", clf)
])

best_est, y_train_pred, train_report, y_val_pred, val_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_val, 
    y_val,
    pipeline_1,
    param_grid,
    n_iter=200,
    search_type='grid'
)

#---------------------------------------------------------
# OPTIMITZEM UMBRAL amb una validació interna del mateix split del train
#---------------------------------------------------------

# Trobem el millor umbral per maximitzar el F1 score
# Calibrem el model per tenir les millors probabilitats calibrades
clf_cal = CalibratedClassifierCV(best_est, method="isotonic", cv=5)

# Reentrenem el model el train
clf_cal.fit(X_train, y_train)

proba_val = clf_cal.predict_proba(X_val)[:,1]

prec, rec, thr = precision_recall_curve(y_val, proba_val)

# Calucl respecte f1, per maximitzar el F1 score
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1)]
print(f"Umbral óptimo en validación: {best_thr:.3f}  ⇒  F1={f1.max():.4f}")


#---------------------------------------------------------
# EVALUEM EL TEST original per evaluar el model
#---------------------------------------------------------
proba_test = clf_cal.predict_proba(X_test)[:,1]
y_test_pred  = (proba_test >= best_thr).astype(int)

print("\n== Classification report en TEST ==")
print(classification_report(y_test, y_test_pred, digits=4))

# Plot de la matriu de confusió
mat_confusio(
    f"UAjust_{model_name}_v1",  
    y_test,
    y_test_pred,
    save='yes'
    )