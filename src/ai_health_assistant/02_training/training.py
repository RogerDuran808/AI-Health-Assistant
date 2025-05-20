import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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

import warnings
warnings.filterwarnings('ignore')


################ Quan tinguem la bona forma de entrenar el model ##################


# Load the dataset
df = pd.read_csv('data/df_preprocessed.csv')

print(f"Shape: {df.shape}")


# Comprovem quina es les estructura de les nostres dades faltants en el target
TARGET = 'TIRED'

df = df.dropna(subset=[TARGET])


# Difinim X i el target y
# Prediccio de TIRED
X = df.drop(columns=[TARGET])
y = df[TARGET]

numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

print(f"\nCol. numeriques ({len(numerical_features)}): \n{numerical_features}")
print(f"Col. categoriques ({len(categorical_features)}): \n{categorical_features}")



# Estratifiquem respecte un dels targets per tal d'assegurar el bon split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


# Define classifiers
CLASSIFIERS = {
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "SVM": SVC(random_state=42, probability=True),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# Param grids pel GridSearchCV
# Complexitat reduida per tal que no porti un temps exegerat de execució
# Un cop trobem el model bó podrem augmenter la complexitat del model
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
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__class_weight": ["balanced", "balanced_subsample"]
    },
    "GradientBoosting": {
        "classifier__n_estimators": [200, 400],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_results = []
base_models = {}


for model, classifier in CLASSIFIERS.items():
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier)
    ])

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS[model],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    gs.fit(X_train, y_train)


    best = gs.best_estimator_
    base_models[model] = best
    
    y_pred = best.predict(X_test)

    report = classification_report(
        y_test,
        y_pred, # Fem un predict amb el millor model trobat i comparem
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    base_results.append({
        "Target":                TARGET,
        "Experiment":            "Entrenament basic",
        "Model":                 model,
        "Best Params":           gs.best_params_,
        "Best CV":               gs.best_score_,
        "Test Precision (1)":    report["1"]["precision"],
        "Test Recall (1)":       report["1"]["recall"],
        "Test F1 (1)":           report["1"]["f1-score"],
        "Test F1 (macro global)": f1_score(y_test, y_pred, average="macro"),
        "Test Accuracy":         report["accuracy"],
    })

    print(f"{model:20s} | Best CV: {gs.best_score_:.4f} | Test F1(cl1): {report["1"]["f1-score"]:.4f} | Acc: {report["accuracy"]:.4f}")