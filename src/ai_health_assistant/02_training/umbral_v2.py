from ai_health_assistant.utils.train_helpers import optimize_threshold, mat_confusio, train_models
import pandas as pd
from sklearn.model_selection import train_test_split
from ai_health_assistant.utils.model_config import get_classifier_config
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report

# Load el dataset
df = pd.read_csv('data/df_preprocessed.csv')

# Comprovem quina es les estructura de les nostres dades faltants en el target
TARGET = 'TIRED'

# Difinim X i el target y
# Prediccio de TIRED
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Fem un X i y temporals que serveixen per fer la divisió anterior del 80% i per tal de separar el 20% amb el qual testejarem
X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

# Fem una altre divisió del 80% per entrenament i per fer la validació
X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42)



# ---------------------------------------------------------
# Definim el model a utilitzar
model_name = "LGBM"  # Opcions: MLP, SVM, RandomForest, GradientBoosting, BalancedRandomForest, LGBM

# Obtenim el classificador i els seus paràmetres des de la configuració centralitzada
clf, param_grid = get_classifier_config(model_name)
# ---------------------------------------------------------

# Definim el balancing method
balancing_method = SMOTETomek(random_state=42)  # Combina oversampling i undersampling

# Fl pipeline amb millors resultats:
pipeline = ImbPipeline([
    ("balancing", balancing_method),
    ("classifier", clf)
])

best_est, y_train_pred, train_report, y_val_pred, val_report, best_params, best_score = train_models(
    X_train, 
    y_train, 
    X_val, 
    y_val,
    pipeline,
    param_grid,
    search_type='grid'
)

threshold = optimize_threshold(best_est, X_val, y_val)
y_pred_optimized = (best_est.predict_proba(X_test)[:, 1] >= threshold).astype(int)

print("\n== Classification report en TEST ==")
print(classification_report(y_test, y_pred_optimized, digits=4))

# Plot de la matriu de confusió
mat_confusio(
    f"UAjust_{model_name}_v2",  
    y_test,
    y_pred_optimized,
    save='yes'
    )