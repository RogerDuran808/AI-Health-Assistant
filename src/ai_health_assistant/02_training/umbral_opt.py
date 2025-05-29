# Agafem el estimador del model que volem ajustar el umbral
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV

from scipy.stats import randint, uniform
from ai_health_assistant.utils.train_helpers import mat_confusio, train_models, optimize_threshold
from sklearn.inspection import permutation_importance


# # Carreguem el model si volem fer servir algun anterior, lo ideal pero es tornar a fer el search sobre el nou split
# model_name = "BalancedRandomForest" # RandomForest, GradientBoosting, MLP, SVM
# model = joblib.load(f'models/{model_name}_TIRED.joblib')

# Load el dataset
df = pd.read_csv('data/df_preprocessed.csv')

# Comprovem quina es les estructura de les nostres dades faltants en el target
TARGET = 'TIRED'

# Difinim X i el target y
# Prediccio de TIRED
X = df.drop(columns=[TARGET])
y = df[TARGET]



# Definim classifiers
CLASSIFIERS = {
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "SVM": SVC(random_state=42, probability=True),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False),
    "LGBM": LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, importance_type='gain', verbose=0)
}
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

    # Els millors parametres trobats els posare al costat per tenir una referencia. Best F1 = 0.65, Acc= 0.59 o F1=0.625 i Acc=0.72 
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

    "LGBM": {
    # "classifier__n_estimators": randint(500, 1001),
    # "classifier__learning_rate": uniform(0.01, 0.1),
    # "classifier__num_leaves": randint(31, 128),
    # "classifier__reg_alpha": uniform(0, 0.5),
    # "classifier__reg_lambda": uniform(0, 1),
    # "classifier__min_child_samples": randint(5, 21),
    # "classifier__subsample": uniform(0.8, 0.2),
    # "classifier__colsample_bytree": uniform(0.8, 0.2)

    # Millors parametrs pel model - LGBM:
    'classifier__colsample_bytree': [0.9116586907214196], 
    'classifier__learning_rate': [0.09826363431893397], 
    'classifier__min_child_samples': [11], 
    'classifier__n_estimators': [508], 
    'classifier__num_leaves': [49], 
    'classifier__reg_alpha': [0.3629778394351197], 
    'classifier__reg_lambda':   [0.8971102599525771], 
    'classifier__subsample': [0.9774172848530235]
    }
}

model_name = "LGBM" # RandomForest, GradientBoosting, MLP, SVM, BalancedRandomForest ...
clf = CLASSIFIERS[model_name]
param_grid = PARAM_GRIDS[model_name]

balancing_method = SMOTETomek(random_state=42)  # Combina oversampling i undersampling

# Farem proves també amb el BalancedRandomForestClassifier
pipeline = ImbPipeline([
    ("balancing", balancing_method),
    ("classifier", clf)
])

##############################################################################
# SPLITS pel calcul del umbral
##############################################################################

# Fem un X i y temporals que serveixen per fer la divisió anterior del 80% i per tal de separar el 20% amb el qual testejarem
X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

# Fem una altre divisió del 80% per entrenament i per fer la validació
X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42)

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

# Fem un print dels paràmetres que ha utilitzat en el cas de fer un search
print('\nParametres utilitzats el model:')
print(best_est)

###########################################################################
# OPTIMITZEM UMBRAL amb una validació interna del mateix split del train
###########################################################################

# Trobem el millor umbral per maximitzar el F1 score
# Calibrem el model per tenir les millors probabilitats calibrades
clf_cal = CalibratedClassifierCV(best_est, method="isotonic", cv=5)

# Reentrenem el model el train
clf_cal.fit(X_train, y_train)

proba_val = clf_cal.predict_proba(X_val)[:,1]

prec, rec, thr = precision_recall_curve(y_val, proba_val)

# # Calucl respecte f1, el problema es que em carrego la classe 0
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1)]
print(f"Umbral óptimo en validación: {best_thr:.3f}  ⇒  F1={f1.max():.4f}")

# Calcul del umbral respecte f1_macro, dona resultats més estables
# f1_macro = [f1_score(y_val, proba_val >= t, average="macro") for t in thr]
# best_thr = thr[np.argmax(f1_macro)]
# print(f"Umbral óptimo (macro-F1): {best_thr:.3f}")

###########################################################################
# EVALUEM EL TEST original per evaluar el model
###########################################################################
proba_test = clf_cal.predict_proba(X_test)[:,1]
y_test_pred  = (proba_test >= best_thr).astype(int)

print("\n== Classification report en TEST ==")
print(classification_report(y_test, y_test_pred, digits=4))

# Plot de la matriu de confusió
mat_confusio(
    f"Umbral Ajustat {model_name}",  
    y_test,
    y_test_pred,
    save='yes'
    )