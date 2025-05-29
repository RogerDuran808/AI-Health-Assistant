import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer, OneHotEncoder, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer


#################################################################################
# Funcions per al preprocessament del dataset netejat
#####################################################################

def feature_engineering(df):
    """ 
    Apliquem feature engineering per tal de crear noves columnes i millorar el model
    Al preprocessament seleccionarem les columnes que realemnt aportin valor
    """
    # --- Feature engineering rápido -------------------------------------------
    df_fe = df.copy()

    df_fe["stress_per_sleep_eff"] = df_fe["stress_score"] / (df_fe["sleep_efficiency"] + 1e-3)
    df_fe["hr_delta"] = df_fe["bpm"] - df_fe["resting_hr"]

    df_fe["steps_norm_cal"] = df_fe["steps"] / (df_fe["calories"] + 1e-3)

    df_fe["wake_after_sleep_pct"] = (
        df_fe["minutesAwake"] /
        (df_fe["minutesAwake"] + df_fe["minutesAsleep"] + 1e-3)
    )

    df_fe["deep_sleep_score"] = df_fe["sleep_deep_ratio"] * df_fe["sleep_efficiency"]


    df = df_fe
    return df

###################### CREEM EL PREPROCESSADOR ######################
def build_preprocessor(numeric_cols, categoric_cols):
    """Crea i retorna el ColumnTransformer que aplica imputacions, transformacions i escalat a continuació.
    """
    numeric_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),  # Mejor imputer para capturar relaciones
        ("transformer", QuantileTransformer(output_distribution='normal')),  # Mejor para distribuciones no normales
        ("scaler", RobustScaler()),  # Más robusto a outliers
    ])

    categoric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categoric_pipe, categoric_cols),
    ])

    return preprocessor

########################### PREPROCESSEM LES DADES ##################################
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

    categoric_cols = X.select_dtypes(exclude=['number']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    preprocessor = build_preprocessor(numeric_cols, categoric_cols)

    X_proc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=df.index)
    df_final = pd.concat([X_proc_df, y], axis=1)
    return df_final



########################## FLUXE DEL PREPROCESSAMENT #############################
def preprocess_data(input_path, output_path, target, features):
    """Flux complet de preprocessament:
    1. Llegeix el CSV (ja netejat).
    2. Aplica transformacions i exporta CSV preprocessat.
    3. Retorna el DataFrame resultant.
    """

    df = pd.read_csv(input_path)
    df = feature_engineering(df)
    df_final = preprocess_dataframe(df, target, features)

    df_final.to_csv(output_path, index=False)
    
    print(f"Dades preprocessades guardades a: {output_path}")

    return df_final


