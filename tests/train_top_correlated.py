"""Prova d'entrenament utilitzant les columnes amb major correlació
amb la variable TIRED.
Aquest script segueix la lògica de ``training.py`` però selecciona
les columnes numèriques amb més correlació.
"""
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from ai_health_assistant.utils.prep_helpers import TARGET, build_preprocessor
from ai_health_assistant.utils.model_config import get_classifier_config, BALANCING_METHODS
from ai_health_assistant.utils.train_helpers import (
    train_models,
    append_results,
    mat_confusio,
    plot_learning_curve,
    save_model,
    update_metrics_file,
)

# ---------------------------------------------------
# Lectura de dades
# ---------------------------------------------------
train_path = 'data/df_engineered_train.csv'
test_path = 'data/df_engineered_test.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# ---------------------------------------------------
# Selecció automàtica de columnes per correlació
# ---------------------------------------------------
correlacions = df_train.corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
# Eliminem la correlació amb la pròpia variable objectiu
correlacions = correlacions.drop(TARGET, errors='ignore')
# Ens quedem amb un màxim de vuit columnes
selected_features = correlacions.head(8).index.tolist()
print(f"Columnes utilitzades: {selected_features}")

# ---------------------------------------------------
# Preparació del train i test
# ---------------------------------------------------
X_train = df_train[selected_features]
y_train = df_train[TARGET]

X_test = df_test[selected_features]
y_test = df_test[TARGET]

# ---------------------------------------------------
# Construïm el preprocessador i definim el model
# ---------------------------------------------------
preprocessor = build_preprocessor(df_train, selected_features)

model_name = 'BalancedRandomForest'
clf, param_grid = get_classifier_config(model_name)

balancing_method = BALANCING_METHODS['SMOTETomek']

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('balancing', balancing_method),
    ('classifier', clf)
])

# ---------------------------------------------------
# Entrenament i registre de resultats
# ---------------------------------------------------
results = []

best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score = train_models(
    X_train,
    y_train,
    X_test,
    y_test,
    pipeline,
    param_grid,
    n_iter=10,
    search_type='random'
)

results_df = append_results(
    results,
    f'{model_name}_correlated',
    train_report,
    test_report,
    best_params,
    best_score
)

mat_confusio(
    f'{model_name}_correlated',
    y_test,
    y_test_pred,
    save='yes'
)

plot_learning_curve(
    f'{model_name}_correlated',
    best_est,
    X_train,
    y_train,
    save='yes'
)

update_metrics_file(results_df)

save_model(best_est, f'{model_name}_correlated', save_external='no')
