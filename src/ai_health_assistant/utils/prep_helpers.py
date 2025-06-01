import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer, OneHotEncoder, RobustScaler, QuantileTransformer, robust_scale
from sklearn.impute import SimpleImputer, KNNImputer


#################################################################################
# Funcions per al preprocessament del dataset netejat
#####################################################################


###################### CREEM EL PREPROCESSADOR ######################
def build_preprocessor(numeric_cols, categoric_cols):
    """Crea i retorna el ColumnTransformer que aplica imputacions, transformacions i escalat a continuació.
    """
    numeric_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),  # KNNImputer per capturar relacions
        ("transformer", QuantileTransformer(output_distribution='normal')),  # QuantileTransformer per distribucions no normal
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


###################### FEATURE ENGINEERING ######################
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


########################### PREPROCESSEM LES DADES ##################################
def preprocess_dataframe(df_train, df_test, target, features):
    """
    Preprocessa els conjunts d'entrenament i prova per separat per evitar data leakage.
    
    Args:
        df_train: DataFrame d'entrenament
        df_test: DataFrame de prova
        target: Nom de la columna objectiu
        features: Llista de característiques a utilitzar
        
    Returns:
        Tupla amb (X_train_processed, X_test_processed, y_train, y_test, preprocessor)
    """
    # Assegurem que només utilitzem les columnes especificades
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()

    y_train = df_train[target].copy()
    y_test = df_test[target].copy()
    
    # Identifiquem columnes numèriques i categòriques
    categoric_cols = X_train.select_dtypes(exclude=['number']).columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    
    # Construïm el preprocessador
    preprocessor = build_preprocessor(numeric_cols, categoric_cols)
    
    # Apliquem el preprocessament als conjunts d'entrenament i prova
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test) # Fem nomes un transform pel test
    
    # Obtenim els noms de les característiques processades
    feature_names = preprocessor.get_feature_names_out()
    
    # Convertim a DataFrames
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor



########################## FLUXE DEL PREPROCESSAMENT #############################
def preprocess_data(train_path, test_path, output_dir, target, features):
    """
    Flux complet de preprocessament per als conjunts d'entrenament i prova:
    1. Llegeix els CSVs d'entrenament i prova
    2. Aplica feature engineering a cada conjunt per separat
    3. Preprocessa els conjunts evitant data leakage
    4. Guarda els resultats i retorna els DataFrames processats
    
    Args:
        train_path: Ruta al fitxer CSV d'entrenament
        test_path: Ruta al fitxer CSV de prova
        output_dir: Directori on es guardaran els resultats
        target: Nom de la columna objectiu
        features: Llista de característiques a utilitzar
        
    Returns:
        Tupla amb (X_train, X_test, y_train, y_test, preprocessor)
    """
    
    # Llegim les dades
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Apliquem feature engineering per separat
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Preprocessem les dades
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataframe(
        train_df, test_df, target, features
    )
    
    # Guardem els resultats
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    df_train.to_csv(f"{output_dir}_train.csv", index=False)
    df_test.to_csv(f"{output_dir}_test.csv", index=False)
    
    print(f"Dades preprocessades guardades a {output_dir}")
    print(f"  - Train: {output_dir}_train.csv")
    print(f"  - Test:  {output_dir}_test.csv")
    
    return df_train, df_test


