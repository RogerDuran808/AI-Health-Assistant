import pandas as pd
import numpy as np
import sys
if sys.prefix != sys.base_prefix:  # Si estem a l'entorn virtual no fa plot, ja que no em funciona
    import matplotlib
    matplotlib.use("Agg") 

import matplotlib.pyplot as plt


from sklearn.model_selection import  GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV

from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

from pathlib import Path
import joblib
import os


############## Entrenament del model ###############################
def train_models(X_train, y_train, X_test, y_test, pipeline, param_grid, scoring = 'f1', cv = 'StratifiedKFold', n_iter = 100, search_type = 'random'):
    '''
    Entrenament del model amb buscador de hiperparàmetres.\n
    
    Arguments: 
    - X_train: features del train
    - y_train: target del train
    - X_test: features del test
    - y_test: target del test
    - pipeline: pipeline almenys amb el preprocessador i el classifier
    - param_grid: parametres per fer el gridsearch (si es un diccionari amb diversos classifiers es pot aplicar com: param_grid[model_name])
    - scoring: per defecte F1 para clase 1.
    - cv: validació creuada, per defecte StratifiedKFold amb 5 splits.
    - n_iter: nombre de iteracions, per defecte 100.
    - search_type: 'random' (per defecte) o es pot fer 'grid' per fer un GridSearchCV
    
    Return:\n
    - best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score
    '''
    if scoring == 'f1':
        scoring = make_scorer(f1_score, pos_label=1)
    if cv == 'StratifiedKFold':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if search_type == 'grid':
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True
        )
    
    print(f"Entrenant model...")
    search.fit(X_train, y_train)
    
    # Millor estimador
    best_est = search.best_estimator_

    # Predicció sobre el train
    y_train_pred = best_est.predict(X_train)

    train_report = classification_report(
        y_train,
        y_train_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    # Predicció sobre el test
    y_test_pred = best_est.predict(X_test)

    test_report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    # Print del report (resumit amb els valors més importants)
    print(f"\nTrain F1 (1): {train_report["1"]["f1-score"]:.4f} | Test F1 (1): {test_report["1"]["f1-score"]:.4f} | Train Acc: {train_report["accuracy"]:.4f} | Test Acc: {test_report["accuracy"]:.4f}")
    print(classification_report(y_test, y_test_pred, digits=4))

    return best_est, y_train_pred, train_report, y_test_pred, test_report,  search.best_params_, search.best_score_


############## Append Results ###############################
def append_results (list_results, model_name, train_report, test_report, best_params, best_score, experiment = None):
    '''
    Crea un **dataframe amb els resultats** de la predicció i el model, fa un append a una llista
    i retorna el dataframe i el guarda el csv a results. Les columnes a poder mostrar son:
    - "Model"
    - "Experiment"
    - "Best Params"
    - "Best CV"
    - "Train F1 (1)"
    - "Train F1 (macro global)"
    - "Train Accuracy"
    - "Test Precision (1)"
    - "Test Recall (1)"
    - "Test F1 (1)"
    - "Test F1 (macro global)"
    - "Test Accuracy"\n

    Arguments:
    - list_results: llista on guardem els resultats
    - model: nom del model
    - train_report: report de la predicció sobre el train
    - test_report: report de la predicció sobre el test
    - best_params: millors parametres trobats
    - best_score: millor score trobat
    - experiment: nom de l'experiment (opcional)\n

    Return:
    - results_df: dataframe amb els resultats
    '''
    if experiment is None:
        experiment = np.nan

    list_results.append({
        "Model":                 model_name,
        "Experiment":            f"{model_name}_{experiment}", # En cas de estar registrant algun experiment
        
        "Best Params":           best_params,
        "Best CV":               best_score,

        "Train F1 (1)":          train_report["1"]["f1-score"],
        "Train F1 (macro global)": train_report["macro avg"]["f1-score"],
        "Train Accuracy":        train_report["accuracy"],

        "Test Precision (1)":    test_report["1"]["precision"],
        "Test Recall (1)":       test_report["1"]["recall"],
        "Test F1 (1)":           test_report["1"]["f1-score"],
        "Test F1 (macro global)": test_report["macro avg"]["f1-score"],
        "Test Accuracy":         test_report["accuracy"],
    })

    results_df = pd.DataFrame(list_results).round(5)

    return results_df

############## Corva d'aprenentatge ###############################
def plot_learning_curve(model_name, best_est, X_train, y_train, save = 'no', score = 'f1'):
    '''
    Genera una corva d'aprenentatge per veure com el model apren sobre el train.\n
    
    Arguments:
    - model_name: nom del model
    - best_est: millor estimador trobat
    - X: features
    - y: target
    - save: per defecte 'no' (mostra la gràfica per notebooks) si es 'yes' guarda la gràfica (no la mostra)
    - score: per defecte 'f1', sino posar quin score es vol utilitzar
    '''
    if score == 'f1':
        scorer = make_scorer(f1_score, pos_label=1)
    else:
        scorer = score

    train_sizes, train_scores, val_scores = learning_curve(
        best_est, X_train, y_train,
        cv=5,
        scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True, 
        random_state=42,
        verbose=0
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    title = f"Corva d'aprenentatge - {model_name}"

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label='Train')
    plt.plot(train_sizes, val_mean,   'o-', label='CV')
    plt.title(title)
    plt.xlabel('Grandària del set')
    plt.ylabel(f'Score ({score})')
    plt.legend()
    plt.grid(True)
    

    if save.lower() == 'yes':
        fname = f"lc_{model_name}.png"
        out_path = f"results/03_training/{fname}"
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Corva d'aprenentatge guardada a: {out_path}")
        plt.close()
        return
    
    # Si no, mostrem la gràfica
    plt.show()

############## Matriu de confusió ###############################
def mat_confusio(title_name, y_true, y_pred, save = 'no'):
    '''
    Matriu de confusió sobre el test, agafant els models registrats en el diccionari. \n
    Guarda la matri de confusió al directori de resultats/03_training
    '''
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1] )
    disp.plot(cmap='Blues')
    plt.title(f"Matriu de confusió - {title_name}")
    
    if save.lower() == 'yes':
        fname = f"cm_{title_name}.png"
        out_path = f'results/03_training/{fname}'
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Confusion matrix guardada a: {out_path}")
        plt.close()
        return
    
    # Si no, mostrem la gràfica
    plt.show()

############## Optimitza l'umbral de decisió ###############################
def optimize_threshold_v1(classifier, X_val, y_val, target_precision=0.5):
    """
    Optimitza l'umbral de decisió (versio 1 per umbral_v1.py) per maximitzar el F1 macro avg mantenint el recall (1) >= target_recall
    """
    y_scores = classifier.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_scores >= threshold).astype(int)
        precision = precision_score(y_val, y_pred, pos_label=1)
        f1 = f1_score(y_val, y_pred)

        if precision >= target_precision and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def optimize_threshold_v2(classifier, X_val, y_val, target_recall=0.7):
    """
    Optimitza l'umbral de decisió (versio 2, forma alternativa per umbral_v2.py) per maximitzar la precisó mantenint el recall >= target_recall
    """
    y_scores = classifier.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_precision = 0
    
    # Probar diferents umbrals
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_scores >= threshold).astype(int)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        # Si el recall cumpleix amb l'objectiu i la precisio es millor que la anterior
        if recall >= target_recall and precision > best_precision:
            best_precision = precision
            best_threshold = threshold
    
    return best_threshold


############### Registre Metriques ###############################
def update_metrics_file(metrics: pd.DataFrame, filename="results/03_training/metrics.csv"):
    columnas = ["Model", "Train F1 (1)", "Train F1 (macro global)", "Train Accuracy", "Test Precision (1)", "Test Recall (1)", "Test F1 (1)", "Test F1 (macro global)", "Test Accuracy", "Best Params"]

    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=columnas)
    
    fila_nova = metrics[columnas].copy()
    model_name = metrics["Model"].iloc[0]
    rewrite = df["Model"] == model_name
    
    if rewrite.any():
        df.loc[rewrite, columnas] = fila_nova.values
    else:
        df = pd.concat([df, fila_nova], ignore_index=True)
    
    df = df.sort_values(by="Test F1 (1)", ascending=False)
    
    df.to_csv(filename, index=False)
    print(f'\nMétriques guardades a {filename}\n')


def update_experiments_file(metrics: pd.DataFrame, filename="../results/02_experiments/experiments.csv"):
    columnas = ["Experiment", "Train F1 (1)", "Train F1 (macro global)", "Train Accuracy", "Test Precision (1)", "Test Recall (1)", "Test F1 (1)", "Test F1 (macro global)", "Test Accuracy", "Best Params"]

    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=columnas)

    for _, fila_nova in metrics[columnas].iterrows():
        experiment_name = fila_nova["Experiment"]
        rewrite = df["Experiment"] == experiment_name
        if rewrite.any():
            for col in columnas:
                df.loc[rewrite, col] = fila_nova[col]
        else:
            df = pd.concat([df, pd.DataFrame([fila_nova])], ignore_index=True)

    df = df.sort_values(by="Test F1 (1)", ascending=False)

    df.to_csv(filename, index=False)
    print(f'\nMétriques guardades a {filename}\n')

############### Guardem Models ###############################
def save_model(best_estimator, model_name, save_external='no'):
    """
    Si save_external = 'yes', desa el model en:
      1) ./models/{model_name}_model.joblib
      2) ../AI-Health-Assistant-WebApp/backend/models/{model_name}_model.joblib
      Per tal de poder-lo utilitzar en la webapp.\n

    En cas de que save_external = 'no', nomes desa el model en:
      1) ./models/{model_name}_model.joblib
    """
    # RUTA DINS DEL REPOSITORI ACTUAL
    local_dir = Path(__file__).parent.parent.parent.parent / "models"
    local_path = local_dir / f"{model_name}_model.joblib"

    joblib.dump(best_estimator, local_path)  # Guardem el model a la ruta local
    print(f"\nModel guardat localment a: {local_path}\n")

    # RUTA EXTERNA A LA WEBAPP
    if save_external.lower() == 'yes':
        # La ruta es relativa a la ubicació de les meves carpetes
        external_dir = (Path(__file__).parent.parent.parent.parent.parent / "AI-Health-Assistant-WebApp" / "backend" / "models")
        external_path = external_dir / f"{model_name}_model.joblib"

        joblib.dump(best_estimator, external_path)  # Guradem el model a la ruta externa, a la aplicació web
        print(f"\nModel guardat externament a: {external_path}\n")