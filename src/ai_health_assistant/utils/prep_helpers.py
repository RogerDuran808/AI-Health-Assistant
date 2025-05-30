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
    """Feature engineering de diferents paràmetres per millorar la prediccio del model"""
    df_fe = df.copy()
    
    # Feature engineering de diferents paràmetres per millorar la prediccio de TIRED
    df_fe["steps_norm_cal"] = df_fe["steps"] / (df_fe["calories"] + 1e-3)
    df_fe["wake_after_sleep_pct"] = df_fe["minutesAwake"] / (df_fe["minutesAwake"] + df_fe["minutesAsleep"] + 1e-3)
    df_fe["deep_sleep_score"] = df_fe["sleep_deep_ratio"] * df_fe["sleep_efficiency"]
    
    # Ratios i proporcions
    df_fe["active_sedentary_ratio"] = (df_fe["very_active_minutes"] + df_fe["moderately_active_minutes"]) / (df_fe["sedentary_minutes"] + 1e-3)
    df_fe["sleep_activity_balance"] = df_fe["minutesAsleep"] / (df_fe["very_active_minutes"] + df_fe["moderately_active_minutes"] + 1e-3)
    
    # Polinomials i combinacions
    df_fe["bmi_hr_interaction"] = df_fe["bmi"] * df_fe["resting_hr"]
    df_fe["sleep_quality_index"] = (df_fe["sleep_deep_ratio"] * 3 + df_fe["sleep_rem_ratio"] * 2) / (df_fe["sleep_wake_ratio"] + 1e-3)
    
    # Variabilitat
    df_fe["hr_zone_variability"] = df_fe[["minutes_below_default_zone_1", "minutes_in_default_zone_1", 
                                          "minutes_in_default_zone_2", "minutes_in_default_zone_3"]].std(axis=1)

    # Ratios más informativos para fatiga
    df_fe["activity_intensity"] = df_fe["very_active_minutes"] * 3 + df_fe["moderately_active_minutes"] * 2 + df_fe["lightly_active_minutes"]
    df_fe["recovery_factor"] = df_fe["minutesAsleep"] / (df_fe["activity_intensity"] + 1e-3)

    # Características polinómicas importantes
    df_fe["sleep_eff_rmssd"] = df_fe["sleep_efficiency"] * df_fe["rmssd"]
    
    # Características cíclicas
    df_fe["active_to_rest_transition"] = df_fe["activity_intensity"] / (df_fe["minutesAsleep"] + df_fe["minutesAwake"] + 1e-3)

    df_fe["active_to_total_ratio"] = (df_fe["very_active_minutes"] + df_fe["moderately_active_minutes"] + df_fe["lightly_active_minutes"]) / (24*60)

    
    return df_fe

###################### CREEM EL PREPROCESSADOR ######################
def build_preprocessor(numeric_cols, categoric_cols):
    """Crea i retorna el ColumnTransformer que aplica imputacions, transformacions i escalat a continuació.
    """
    numeric_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights='distance')),  # KNNImputer per capturar relacions
        ("transformer", QuantileTransformer(output_distribution='normal', n_quantiles=1000)),  # QuantileTransformer per distribucions no normal      s
        ("scaler", RobustScaler()),  # RobustScaler per a outliers
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


