import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # no interactiu sino causa err
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

def train_models(X_train, y_train, X_test, y_test, pipeline, param_grid, scoring = None, cv = None, n_iter = 100):
    '''
    Entrenament del model amb buscador de hiperparàmetres, en el cas de no posar scoring es fa
    un search a f1 score de la classe 1. En cas de on posar cv es fa un StratifiedKFold. I 100 nombre de iteracions\n

    La funció retorna:\n
    - best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score
    '''
    if scoring is None:
        scoring = make_scorer(f1_score, pos_label=1)
  
    if cv is None:
        cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    return best_est, y_train_pred, train_report, y_test_pred, test_report,  search.best_params_, search.best_score_



def append_results (list_results, model, train_report, test_report, best_params, best_score):
    '''
    Crea un **dataframe amb els resultats** de la predicció i el model, fa un append a una llista
    i retorna el dataframe i el guarda el csv a results. Les columnes a poder mostrar son:\n
        - "Model"
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
    
    TARGET= "TIRED"
    list_results.append({
        "Target":                TARGET,
        "Model":                 model,
        
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

    print(f"{model:20s} | Train F1 (1): {train_report["1"]["f1-score"]:.4f} | Test F1 (1): {test_report["1"]["f1-score"]:.4f} | Train Acc: {train_report["accuracy"]:.4f} | Test Acc: {test_report["accuracy"]:.4f}")

    results_df = pd.DataFrame(list_results)
    print('\n')
    print(results_df)
    results_df.to_csv(f'results/02_training/training_results.csv', index=False)

    return results_df


def plot_learning_curve(model_name, dict_models, X, y):
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
    plt.show()

    fname = f"learning_curve_{model_name}.png"
    plt.savefig(f'results/02_training/{fname}', bbox_inches='tight')
    plt.close()
    print(f"Corva d'aprenentatge guardada a: results/02_training/{fname}")


def mat_confusio(title_name, y_true, y_pred):
    '''
    Matriu de confusió sobre el test, agafant els models registrats en el diccionari. \n
    Guarda la matri de confusió al directori de resultats/02_training
    '''
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1] )
    disp.plot(cmap='Blues')
    plt.title(f"Matriu de confusió - {title_name}")
    plt.show()
    fname = f"confusion_matrix_{title_name}.png"
    plt.savefig(f'results/02_training/{fname}', bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix guardada a: results/02_training/{fname}")
