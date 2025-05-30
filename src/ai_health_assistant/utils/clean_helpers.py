import pandas as pd
import numpy as np

########################################################################################
# Funcions per neteja de dades Fitbit LifeSnaps
#####################################################################################

def drop_irrelevant(df):
    # Eliminem les columnes irrellevants per la predicció
    cols_irr = [
        'Unnamed: 0', 'id', 'date',
        'mindfulness_session', 'step_goal', 'step_goal_label', 'activityType', 'badgeType', 'min_goal', 'max_goal',
        'filteredDemographicVO2Max', 'exertion_points_percentage', 'responsiveness_points_percentage', 'distance', 'scl_avg', 'sleep_duration',
        'ENTERTAINMENT', 'GYM', 'HOME', 'HOME_OFFICE', 'OTHER', 'OUTDOORS', 'TRANSIT', 'WORK/SCHOOL'
    ]
    return df.drop(columns=cols_irr, errors='ignore')


def fix_bmi(df):
    df = df.copy()
    # Fem una correcció de la variabe del bmi
    limits = (0, 18.5, 24.9, 29.9, 34.9) # limits segons les dades que tenim i el bmi
    categories_BMI = [
        'Infrapes',
        'Normal',
        'Sobrepes',
        'Obes'
    ]
    # Com que els valors >=25 es troben igualment a la categoria de Sobrepes, els imputarem com a 25
    df.loc[df['bmi'] == '>=25', 'bmi'] = 25

    # Assignar a 18.4 els valors '<19' perquè caiguin dins 'Infrapes'
    df.loc[df['bmi'] == '<19', 'bmi'] = 18.4

    # Assignar a 30 els valors '>=30' perquè caiguin dins 'Obes'
    # Suposarem que no hi ha valors superiors a 35 i per tant no hi haura cap més categoria de bmi
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
def handle_outliers_advanced(df):
    """Trata outliers considerando la importancia de características para el target"""
    df_copy = df.copy()
    
    # Calculamos correlación para determinar importancia
    corr_with_target = df_copy.corr()[target].abs().sort_values(ascending=False)
    
    # Tratamos outliers de forma diferenciada según importancia
    for col in df_copy.select_dtypes(include=['number']).columns:
        if col != target:
            importance = corr_with_target.get(col, 0)
            # Para características más importantes, tratamiento más conservador
            iqr_factor = 2.0 if importance > 0.2 else 1.5
            
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            # Aplicamos winsorizing según importancia
            df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
    return df_copy


################## DROP COLUMNES segons EDA ########################
def drop_additional_columns(df):
    """
    Elimina columnes addicionals basades en resultats d'EDA.
    """
    cols_to_drop = [
        'ALERT', 'HAPPY', 'NEUTRAL', 'SAD', 'RESTED/RELAXED', 'TENSE/ANXIOUS',
        'sleep_points_percentage'
    ]
    return df.drop(columns=cols_to_drop, errors='ignore')

def correct_columns(df):
    """
    Corregim les columne necessaries afectades:
    - sleep_efficiency, ja que esta calculada a partir de les altres i es pot reajustar el calcul.
    """
    # Corregim sleep_efficiency
    df['sleep_efficiency'] = (df['minutesAsleep'] / (df['minutesAsleep'] + df['minutesAwake'])) * 100
    return df

######################### NETEJA DE DADES ###############################

def clean_data(input_path, output_path):
    """
    Flux complet de neteja:
      1) carrega dades
      2) desfés les columnes irrellevants
      3) corregeix BMI
      4) retalla outliers
      5) descarta columnes addicionals
      6) escriu CSV net
    Retorna el DataFrame net.
    """
    df = pd.read_csv(input_path)
    df = drop_irrelevant(df)
    df = fix_bmi(df)
    df = handle_outliers_advanced(df)
    df = drop_additional_columns(df)
    df = correct_columns(df)
    df.to_csv(output_path, index=False)
    
    print(f"Dades netejades exportades a: {output_path}")
    return df

