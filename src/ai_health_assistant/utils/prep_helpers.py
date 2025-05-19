import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer


# =============================================================================
# preprocess_helpers.py
# Funcions utilitàries per al preprocessament del dataset netejat
# =============================================================================

def build_preprocessor(numeric_cols, categoric_cols):
    """Crea i retorna el ColumnTransformer que aplica imputacions, transformacions
    energètiques (PowerTransformer) i escalat a continuació.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scaler", StandardScaler()),
    ])

    categoric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(categories=[["Infrapes", "Normal", "Sobrepes", "Obes"]]),
        ),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categoric_pipe, categoric_cols),
    ])


def preprocess_dataframe(df, target, features):
    """Rep un DataFrame ja net (df_cleaned), aplica transformacions i retorna un
    nou DataFrame preparat (df_preprocessed).
    """
    # Selecció de columnes
    df = df[features + [target]].copy()
    df.dropna(subset=[target], inplace=True)

    # Separació X/y
    y = df[target]
    X = df.drop(columns=[target])

    categoric_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categoric_cols]

    preprocessor = build_preprocessor(numeric_cols, categoric_cols)

    X_proc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=df.index)
    df_final = pd.concat([X_proc_df, y], axis=1)
    return df_final


def preprocess_data(input_path, output_path) -> pd.DataFrame:
    """Flux complet de preprocessament:
    1. Llegeix el CSV (ja netejat).
    2. Aplica transformacions i exporta CSV preprocessat.
    3. Retorna el DataFrame resultant.
    """
    target = "TIRED"
    features = [
        "steps",
        "calories",
        "bpm",
        "sedentary_minutes",
        "resting_hr",
        "minutesAsleep",
        "bmi_tipo",
    ]

    df = pd.read_csv(input_path)
    df_final = preprocess_dataframe(df, target, features)

    # Crea carpeta si no existeix
    out_path = Path(output_path)
    df_final.to_csv(out_path, index=False)
    print(f"Dades preprocessades guardades a: {output_path}")

    return df_final

