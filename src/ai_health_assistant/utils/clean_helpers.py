import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

########################################################################################
# Funcions per neteja de dades Fitbit LifeSnaps
#####################################################################################

def features_split(df, target, features):
    """Selecciona les columnes disponibles del dispositiu fitbit i fem el train test split"""
    df = df[features + [target]].copy()
    # Assegurar-nos que no hi hagi NaN en el target abans de fer el split
    df = df.dropna(subset=[target])

    x_cols = df.drop(columns=[target]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        df[x_cols],
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target]
    )
    return X_train, X_test, y_train, y_test


def fix_bmi(df):
    # Fem una correcció de la variable del bmi
    limits = (0, 18.5, 24.9, 29.9, 34.9)  # límits segons les dades que tenim i el bmi
    categories_BMI = [
        'Infrapes',
        'Normal',
        'Sobrepes',
        'Obes'
    ]
    # Com que els valors >=25 es troben igualment a la categoria de Sobrepes, els imputem com a 25
    df.loc[df['bmi'] == '>=25', 'bmi'] = 25

    # Assignar a 18.4 els valors '<19' perquè caiguin dins 'Infrapes'
    df.loc[df['bmi'] == '<19', 'bmi'] = 18.4

    # Assignar a 30 els valors '>=30' perquè caiguin dins 'Obes'
    # Suposarem que no hi ha valors superiors a 35 i, per tant, no hi haurà cap més categoria de bmi
    df.loc[df['bmi'] == '>=30', 'bmi'] = 30

    # Convertim a float
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

    df['bmi_tipo'] = pd.cut(
        df['bmi'],
        bins=limits,
        labels=categories_BMI,
        right=False
    )
    return df

################### CORRECCIÓ DE OUTILIERS ######################
def handle_outliers(X_train, X_test, multiplier=1.5):
    """Mètode per tractar els outliers utilitzant IQR per cada grup de classe"""

    numeric_cols = X_train.select_dtypes(include="number").columns

    # Compute bounds on training set only
    bounds = {}
    for col in numeric_cols:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)

    # Apply clipping on copies to avoid side-effects
    X_train_cap = X_train.copy()
    X_test_cap = X_test.copy()

    for col, (lower, upper) in bounds.items():
        X_train_cap[col] = X_train_cap[col].clip(lower=lower, upper=upper)
        X_test_cap[col] = X_test_cap[col].clip(lower=lower, upper=upper)

    return X_train_cap, X_test_cap


######################### NETEJA DE DADES ###############################

def clean_data(input_path, output_path, target, features):
    """
    Flux complet de neteja:
      1) carrega dades
      2) selecciona les columnes disponibles al fitbit inspire 3
      3) corregeix BMI
      4) retalla outliers
      5) corregim les columnes necessàries
      6) escriu CSV net
    Retorna els DataFrames netejats de train i test, i el concatenat.
    """
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = features_split(df, target, features)
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
