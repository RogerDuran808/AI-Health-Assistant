"""
Helpers per a la preparació de les dades per a la predicció. Inclou funcions per a la transformació, imputació i escalat de valors.
Incorpora també l'enginyeria de característiques i la selecció d'atributs per a la predicció.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer, OneHotEncoder, RobustScaler, QuantileTransformer, robust_scale
from sklearn.impute import SimpleImputer, KNNImputer

TARGET = 'TIRED'

# Columnes identificadores i data
COLS_ID = ['id', 'date']

# Llista de característiques per a la predicció
FEATURES = [
    "age",
    "gender",
    "bmi",
    # "bmi_tipo",
    "calories",
    "steps",
    "lightly_active_minutes",
    "moderately_active_minutes",
    "very_active_minutes",
    "sedentary_minutes",
    "resting_hr",
    "minutes_below_default_zone_1",
    "minutes_in_default_zone_1",
    "minutes_in_default_zone_2",
    "minutes_in_default_zone_3",
    # "minutesToFallAsleep",
    "minutesAsleep",
    "minutesAwake",
    # "minutesAfterWakeup",
    "sleep_efficiency",
    "sleep_deep_ratio",
    "sleep_light_ratio",
    "sleep_rem_ratio",
    "sleep_wake_ratio",
    "daily_temperature_variation",
    "rmssd",
    "spo2",
    "full_sleep_breathing_rate",
    
    # Enginyeria de característiques: noves columnes
    'wake_after_sleep_pct',
    'steps_norm_cal',
    'deep_sleep_score',
    'active_sedentary_ratio',
    'sleep_activity_balance',
    'bmi_hr_interaction',
    'sleep_quality_index',
    'hr_zone_variability',
    'recovery_factor',
    'sleep_eff_rmssd',
    'active_to_rest_transition',
    'active_to_total_ratio'
]

COLUMNES_DATASET = COLS_ID + FEATURES

#################################################################################
# Funcions per al preprocessament del dataset netejat
#################################################################################

###################### ENGINYERIA DE CARACTERÍSTIQUES ######################
def feature_engineering(df):
    """Genera noves característiques per millorar la predicció del nivell de cansament."""
    df_fe = df.copy()
    
    # Enginyeria de característiques per optimitzar la predicció de TIRED
    df_fe["steps_norm_cal"] = df_fe["steps"] / (df_fe["calories"] + 1e-3)
    df_fe["wake_after_sleep_pct"] = df_fe["minutesAwake"] / (df_fe["minutesAwake"] + df_fe["minutesAsleep"] + 1e-3)
    df_fe["deep_sleep_score"] = df_fe["sleep_deep_ratio"] * df_fe["sleep_efficiency"]
    
    # Càlcul de ràtios i proporcions
    df_fe["active_sedentary_ratio"] = (df_fe["very_active_minutes"] + df_fe["moderately_active_minutes"]) / (df_fe["sedentary_minutes"] + 1e-3)
    df_fe["sleep_activity_balance"] = df_fe["minutesAsleep"] / (df_fe["very_active_minutes"] + df_fe["moderately_active_minutes"] + 1e-3)
    
    # Combinacions i polinomials
    df_fe["bmi_hr_interaction"] = df_fe["bmi"] * df_fe["resting_hr"]
    df_fe["sleep_quality_index"] = (df_fe["sleep_deep_ratio"] * 3 + df_fe["sleep_rem_ratio"] * 2) / (df_fe["sleep_wake_ratio"] + 1e-3)
    
    # Càlcul de la variabilitat de zones de FC
    df_fe["hr_zone_variability"] = df_fe[[
        "minutes_below_default_zone_1",
        "minutes_in_default_zone_1",
        "minutes_in_default_zone_2",
        "minutes_in_default_zone_3"
    ]].std(axis=1)

    # Ràtios més informatius de fatiga
    df_fe["activity_intensity"] = (
        df_fe["very_active_minutes"] * 3
        + df_fe["moderately_active_minutes"] * 2
        + df_fe["lightly_active_minutes"]
    )
    df_fe["recovery_factor"] = df_fe["minutesAsleep"] / (df_fe["activity_intensity"] + 1e-3)
    
    # Interacció entre eficiència de son i RMSSD
    df_fe["sleep_eff_rmssd"] = df_fe["sleep_efficiency"] * df_fe["rmssd"]
    
    # Transició activitat/repos i proporció d'activitat total
    df_fe["active_to_rest_transition"] = df_fe["activity_intensity"] / (df_fe["minutesAsleep"] + df_fe["minutesAwake"] + 1e-3)
    df_fe["active_to_total_ratio"] = (
        df_fe["very_active_minutes"]
        + df_fe["moderately_active_minutes"]
        + df_fe["lightly_active_minutes"]
    ) / (24 * 60)

    return df_fe


###################### CREACIÓ DEL PREPROCESSADOR ######################
def build_preprocessor(df, features):
    """
    Crea i retorna un objecte ColumnTransformer que realitza la imputació, les transformacions i l'escalat de dades.
    Rep com a arguments un DataFrame i la llista de característiques a utilitzar i retorna el preprocessador configurat.
    """
    X_train = df[features].copy()

    numeric_cols = X_train.select_dtypes(include=['number']).columns
    categoric_cols = X_train.select_dtypes(exclude=['number']).columns

    # Pipeline per a variables numèriques amb imputació, transformació i escalat
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Imputació amb mitjana
        ("transformer", QuantileTransformer(output_distribution='normal', random_state=42)),  # Normalització de distribucions
        ("scaler", RobustScaler()),  # Escalat robust davant outliers
    ])

    # Pipeline per a variables categòriques amb imputació i codificació one-hot
    categoric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Imputació amb la modalitat més freqüent
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificació one-hot
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categoric_pipe, categoric_cols),
    ])

    return preprocessor


########################## FLUX DE PREPROCESSAMENT #############################
def preprocess_data(train_path, test_path, output_dir, features, target):
    """
    Preprocessa els conjunts d'entrenament i test per a la predicció. Realitza la càrrega dels fitxers CSV, aplica la generació de característiques de manera independent i desa els resultats.
    Desa els fitxers CSV amb les dades preparades i retorna els DataFrames corresponents.
    """
    # Carrega les dades d'entrenament i de test
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Aplica la generació de característiques a cada conjunt
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)

    # Filtra les columnes de predicció incloent la variable resposta
    df_train = df_train[features + [target]]
    df_test = df_test[features + [target]]
    
    # Desa els fitxers CSV amb les dades netes i les noves característiques
    df_train.to_csv(f"{output_dir}_train.csv", index=False)
    df_test.to_csv(f"{output_dir}_test.csv", index=False)
    
    print(f"Dades netes i amb enginyeria de característiques guardades a {output_dir}")
    print(f"  - Train: {output_dir}_train.csv")
    print(f"  - Test:  {output_dir}_test.csv")
    
    return df_train, df_test
