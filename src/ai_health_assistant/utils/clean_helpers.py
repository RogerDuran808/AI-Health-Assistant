import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET = 'TIRED'

# Columnes disponibles al Fitbit Inspire 3
FEATURES = [
    # Per personalitzar la imputació i aplicar tècniques més eficients
        "id", 
        "date",

        # Característiques
        "age",
        "gender",
        "bmi",
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
        "minutesToFallAsleep",
        "minutesAsleep",
        "minutesAwake",
        "minutesAfterWakeup",
        "sleep_efficiency",
        "sleep_deep_ratio",
        "sleep_light_ratio",
        "sleep_rem_ratio",
        "sleep_wake_ratio",
        "daily_temperature_variation",
        "rmssd",
        "spo2",
        "full_sleep_breathing_rate",
    ]
    

########################################################################################
# Funcions de neteja de dades de Fitbit LifeSnaps
#####################################################################################

def features_split(df, target, features):
    df = df[features + [target]].copy()
    # Eliminar les files amb valors NaN abans del split
    df = df.dropna(subset=[target])

    x_cols = df.drop(columns=[target]).columns.tolist()

    # Fer el split per evitar la fuga d'informació (data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        df[x_cols],
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target]
    )
    return X_train, X_test, y_train, y_test

def drop_no_factible(df):
    # Corregir el conjunt de dades
    # Definir els rangs permesos
    rangs_possibles = {
        "calories":                      {"min": 0, "max": 20000},
        "steps":                         {"min": 0, "max": 150000},
        "lightly_active_minutes":       {"min": None, "max": 1440},
        "moderately_active_minutes":    {"min": None, "max": 1440},
        "very_active_minutes":          {"min": None, "max": 1440},
        "sedentary_minutes":            {"min": None, "max": 1440},
        "resting_hr":                   {"min": 20, "max": 250},
        "minutes_below_default_zone_1": {"min": None, "max": 1440},
        "minutes_in_default_zone_1":    {"min": None, "max": 1440},
        "minutes_in_default_zone_2":    {"min": None, "max": 1440},
        "minutes_in_default_zone_3":    {"min": None, "max": 1440},
        "minutesToFallAsleep":          {"min": None, "max": 1440},
        "minutesAsleep":                {"min": None, "max": 1440},
        "minutesAwake":                 {"min": None, "max": 1440},
        "minutesAfterWakeup":           {"min": None, "max": 1440},
        "sleep_efficiency":             {"min": 0, "max": 100},
        "sleep_deep_ratio":             {"min": 0, "max": 1},
        "sleep_light_ratio":            {"min": 0, "max": 1},
        "sleep_rem_ratio":              {"min": 0, "max": 1},
        "sleep_wake_ratio":             {"min": 0, "max": 1},
        "daily_temperature_variation":  {"min": -15, "max": 15},
        "rmssd":                        {"min": 0, "max": 500},
        "spo2":                         {"min": 50, "max": 100},
        "full_sleep_breathing_rate":    {"min": 1, "max": 60},
    }
    # Neteja per columnes
    for col, valor in rangs_possibles.items():
        # Assignar NaN als valors per sota del mínim
        if valor['min'] is not None:
            df.loc[df[col] < valor['min'], col] = np.nan
        # Assignar NaN als valors per sobre del màxim
        if valor['max'] is not None:
            df.loc[df[col] > valor['max'], col] = np.nan
    return df

def fix_bmi(df):
    # Corregir la variable BMI
    limits = (0, 18.5, 24.9, 29.9, 34.9)  # Límits basats en les categories de BMI
    categories_BMI = [
        'Infrapes',
        'Normal',
        'Sobrepes',
        'Obes'
    ]
    # Com que els valors ≥25 pertanyen a 'Sobrepes', s'imputen amb 25
    df.loc[df['bmi'] == '>=25', 'bmi'] = 25

    # Imputar 18.4 als valors '<19' per classificar-los com 'Infrapes'
    df.loc[df['bmi'] == '<19', 'bmi'] = 18.4

    # Imputar 30 als valors '≥30' per classificar-los com 'Obes'
    # S'assumeix l'absència de valors >35, per la qual cosa no s'afegeixen més categories de BMI
    df.loc[df['bmi'] == '>=30', 'bmi'] = 30

    # Convertir la columna BMI a float
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

    df['bmi_tipo'] = pd.cut(
        df['bmi'],
        bins=limits,
        labels=categories_BMI,
        right=False
    )
    return df

################### CORRECCIÓ D'OUTILIERS ######################
def handle_outliers(X_train, X_test, multiplier=1.5):
    """
    Mètode per tractar els outliers mitjançant l'IQR segons cada classe.

    Aquest mètode calcula els límits inferior i superior en les dades d'entrenament
    i els aplica tant al conjunt d'entrenament com de prova.
    """

    numeric_cols = X_train.select_dtypes(include="number").columns

    # Límits només del conjunt d'entrenament per evitar la fuga d'informació
    bounds = {}
    for col in numeric_cols:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)

    X_train = X_train.copy()
    X_test = X_test.copy()

    for col, (lower, upper) in bounds.items():
        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)

    return X_train, X_test

######################### NETEJA DE DADES ###############################
def clean_data(input_path, output_path, target, features):
    """
    Flux complet de neteja:
      1) Carrega dades.
      2) Selecciona columnes disponibles al Fitbit Inspire 3 i realitza el split.
      3) Elimina valors no factibles.
      4) Corregir la variable BMI.
      5) Tractar outliers.
      6) Escriure els CSV netejats.
    Retorna els DataFrames de train, test i el conjunt complet.
    """
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = features_split(df, target, features)
    X_train = drop_no_factible(X_train)
    X_test = drop_no_factible(X_test)
    X_train = fix_bmi(X_train)
    X_test = fix_bmi(X_test)
    X_train, X_test = handle_outliers(X_train, X_test)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train.to_csv(f'{output_path}_train.csv', index=False)
    df_test.to_csv(f'{output_path}_test.csv', index=False)

    print(f"Dades netejades guardades a {output_path}")
    print(f"  - Train: {output_path}_train.csv")
    print(f"  - Test:  {output_path}_test.csv")
    
    return df_train, df_test

