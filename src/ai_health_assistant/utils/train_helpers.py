import pandas as pd
import numpy as np
import sys
if sys.prefix != sys.base_prefix:  # Si estem a l'entorn virtual no fa plot, ja que no em funciona
    import matplotlib
    matplotlib.use("Agg") 

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from pathlib import Path
import joblib



def train_models(X_train, y_train, X_test, y_test, pipeline, param_grid, scoring = None, cv = None, n_iter = 100, search_type = 'random'):
    '''
    Entrenament del model amb buscador de hiperparàmetres.\n
    Paràmetres:
    - scoring: per defecte F1 para clase 1.
    - cv: validació creuada, per defecte StratifiedKFold amb 5 splits.
    - n_iter: nombre de iteracions, per defecte 100.
    - search_type: 'random' (per defecte) o 'grid' per fer un GridSearchCV

    Si a paramgrid s'utilitza un diccionari amb diversos classifiers es pot aplicar com: param_grid[model_name]\n
    La funció retorna:\n
    - best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score
    '''
    if scoring is None:
        scoring = make_scorer(f1_score, pos_label=1)
    if cv is None:
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

    print(f"\nTrain F1 (1): {train_report["1"]["f1-score"]:.4f} | Test F1 (1): {test_report["1"]["f1-score"]:.4f} | Train Acc: {train_report["accuracy"]:.4f} | Test Acc: {test_report["accuracy"]:.4f}")
    print(classification_report(y_test, y_test_pred, digits=4))

    return best_est, y_train_pred, train_report, y_test_pred, test_report,  search.best_params_, search.best_score_



def append_results (list_results, model, train_report, test_report, best_params, best_score, experiment = None):
    '''
    Crea un **dataframe amb els resultats** de la predicció i el model, fa un append a una llista
    i retorna el dataframe i el guarda el csv a results. Les columnes a poder mostrar son:\n
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
        - "Test Accuracy"
    '''
    if experiment is None:
        experiment = np.nan

    TARGET = "TIRED"
    list_results.append({
        "Target":                TARGET,
        "Model":                 model,
        "Experiment":            experiment, # En cas de estar registrant algun experiment
        
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

    results_df = pd.DataFrame(list_results)

    return results_df


def plot_learning_curve(model_name, dict_models, X, y, save = 'no'):
    estimator = dict_models[model_name]
    f1_cls1 = make_scorer(f1_score, pos_label=1)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=5,
        scoring=f1_cls1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True, random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    title = f"Corva d'aprenentatge - {model_name}"

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label='Train F1')
    plt.plot(train_sizes, val_mean,   'o-', label='CV F1')
    plt.title(title)
    plt.xlabel('Grandària del set')
    plt.ylabel('F1-1')
    plt.legend()
    plt.grid(True)
    

    if save.lower() == 'yes':
        fname = f"learning_curve_{model_name}.png"
        out_path = f"results/02_training/{fname}"
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Corva d'aprenentatge guardada a: {out_path}")
        plt.close()
        return
    
    # Si no, mostrem la gràfica
    plt.show()


def mat_confusio(title_name, y_true, y_pred, save = 'no'):
    '''
    Matriu de confusió sobre el test, agafant els models registrats en el diccionari. \n
    Guarda la matri de confusió al directori de resultats/02_training
    '''
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1] )
    disp.plot(cmap='Blues')
    plt.title(f"Matriu de confusió - {title_name}")
    
    if save.lower() == 'yes':
        fname = f"confusion_matrix_{title_name}.png"
        out_path = f'results/02_training/{fname}'
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Confusion matrix guardada a: {out_path}")
        plt.close()
        return
    
    # Si no, mostrem la gràfica
    plt.show()

def optimize_threshold(classifier, X_val, y_val, target_recall=0.7):
    """
    Optimitz l'umbral de decisió per maximitzar la precisó mantenint el recall >= target_recall
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


def save_model(best_estimator, model_name, save_external='no'):
    """
    Desa el model en:
      1) ./models/{model_name}_TIRED.joblib
      2) ../AI-Health-Assistant-WebApp/backend/models/{model_name}_TIRED.joblib
      Per tal de poder-lo utilitzar en la webapp.
    """
    # RUTA DINS DEL REPOSITORI ACTUAL
    local_dir = Path(__file__).parent.parent.parent.parent / "models"
    local_path = local_dir / f"{model_name}_TIRED.joblib"

    joblib.dump(best_estimator, local_path)  # Guardem el model a la ruta local
    print(f"Model guardat localment a: {local_path}")

    # RUTA EXTERNA A LA WEBAPP
    if save_external.lower() == 'yes':
        # La ruta es relativa a la ubicació de les meves carpetes
        external_dir = (Path(__file__).parent.parent.parent.parent.parent / "AI-Health-Assistant-WebApp" / "backend" / "models")
        external_path = external_dir / f"{model_name}_TIRED.joblib"

        joblib.dump(best_estimator, external_path)  # Guradem el model a la ruta externa, a la aplicació web
        print(f"Model guardat externament a: {external_path}")