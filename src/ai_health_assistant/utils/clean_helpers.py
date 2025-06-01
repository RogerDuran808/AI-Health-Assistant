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


def fix_bmi(X_train, X_test):
    # Fem una correcció de la variable del bmi
    limits = (0, 18.5, 24.9, 29.9, 34.9)  # límits segons les dades que tenim i el bmi
    categories_BMI = [
        'Infrapes',
        'Normal',
        'Sobrepes',
        'Obes'
    ]
    # Com que els valors >=25 es troben igualment a la categoria de Sobrepes, els imputem com a 25
    X_train.loc[X_train['bmi'] == '>=25', 'bmi'] = 25
    X_test.loc[X_test['bmi'] == '>=25', 'bmi'] = 25

    # Assignar a 18.4 els valors '<19' perquè caiguin dins 'Infrapes'
    X_train.loc[X_train['bmi'] == '<19', 'bmi'] = 18.4
    X_test.loc[X_test['bmi'] == '<19', 'bmi'] = 18.4

    # Assignar a 30 els valors '>=30' perquè caiguin dins 'Obes'
    # Suposarem que no hi ha valors superiors a 35 i, per tant, no hi haurà cap més categoria de bmi
    X_train.loc[X_train['bmi'] == '>=30', 'bmi'] = 30
    X_test.loc[X_test['bmi'] == '>=30', 'bmi'] = 30

    # Convertim a float
    X_train['bmi'] = pd.to_numeric(X_train['bmi'], errors='coerce')
    X_test['bmi'] = pd.to_numeric(X_test['bmi'], errors='coerce')

    X_train['bmi_tipo'] = pd.cut(
        X_train['bmi'],
        bins=limits,
        labels=categories_BMI,
        right=False
    )
    X_test['bmi_tipo'] = pd.cut(
        X_test['bmi'],
        bins=limits,
        labels=categories_BMI,
        right=False
    )
    return X_train, X_test

################### CORRECCIÓ DE OUTILIERS ######################
def handle_outliers(X_train, y_train, target):
    """Mètode per tractar els outliers utilitzant IQR per cada grup de classe"""

    for label in [0, 1]:  # Per cada classe
        subset = X_train[y_train == label]

        for col in X_train.select_dtypes(include=['number']).columns:
            if col != target:
                Q1 = subset[col].quantile(0.25)
                Q3 = subset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 0 * IQR
                upper_bound = Q3 + 0 * IQR

                # Aplicar límits de forma específica per cada classe
                X_train.loc[(y_train == label) & (X_train[col] < lower_bound), col] = lower_bound
                X_train.loc[(y_train == label) & (X_train[col] > upper_bound), col] = upper_bound

    return X_train


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
    X_train, X_test = fix_bmi(X_train, X_test)
    X_train = handle_outliers(X_train, y_train, target)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df_full = pd.concat([df_train, df_test], axis=0)

    df_train.to_csv(f'{output_path}_train.csv', index=False)
    df_test.to_csv(f'{output_path}_test.csv', index=False)
    df_full.to_csv(f'{output_path}.csv', index=False)

    print(f"Dades netejades exportades a: {output_path}")
    
    return df_train, df_test, df_full
