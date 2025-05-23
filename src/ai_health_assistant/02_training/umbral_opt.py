# Agafem el estimador del model que volem ajustar el umbral
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, make_scorer

from ai_health_assistant.utils.train_helpers import mat_confusio



# Carreguem el model
model = joblib.load('models/BalancedRandomForest_TIRED.joblib')

# Load el dataset
df = pd.read_csv('data/df_preprocessed.csv')

# Comprovem quina es les estructura de les nostres dades faltants en el target
TARGET = 'TIRED'

# Difinim X i el target y
# Prediccio de TIRED
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Estratifiquem respecte un dels targets per tal d'assegurar el bon split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



###########################################################################
# OPTIMITZEM UMBRAL amb una validació interna del mateix split del train
###########################################################################

# Fem un split petit (10-15%) del train per tal de fer la validació
X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# Reentrenem el model sobre X_tr_sub
model.fit(X_tr_sub, y_tr_sub)

# Trobem el millor umbral per maximitzar el F1 score
proba_val = model.predict_proba(X_val)[:,1]

prec, rec, thr = precision_recall_curve(y_val, proba_val)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1)]

print(f"Umbral óptimo en validación: {best_thr:.3f}  ⇒  F1={f1.max():.4f}")

###########################################################################
# EVALUEM EL TEST original per evaluar el model
###########################################################################
proba_test = model.predict_proba(X_test)[:,1]
y_test_pred  = (proba_test >= best_thr).astype(int)

print("\n== Classification report en TEST ==")
print(classification_report(y_test, y_test_pred, digits=4))

# Plot de la matriu de confusió
mat_confusio(
    "Umbral Ajustat BalancedRF",  
    y_test,
    y_test_pred
    )