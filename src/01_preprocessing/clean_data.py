# Import de llibreries necessàries
import pandas as pd
import numpy as np

########################## Neteja de les dades ################################
'''
Justificació de la neteja de dades al notebook: 01_LifeSnaps_EDA.ipynb
L'objectiu es tenir les dades netes i imputades de forma basica per fer un bon preprocessament

Eliminem les columnes irrellevants per la nostre predicció.
Corregim la variable del bmi i afegim categories de bmi_tipo
Corregim els valors anòmals del dataset

'''
# Importar el fitxer CSV

df = pd.read_csv('data/daily_fitbit_sema_df_unprocessed.csv')

# Eliminem les columnes que no necessitem:
# Motius: 
# - dades que podem obtenir a traves del nostre fitbit inspire 3
# - columnes irrellevants que no aporten valor

df.drop(columns=['Unnamed: 0', 'id', 'date'], inplace=True)
df.drop(columns=['mindfulness_session', 'step_goal', 'step_goal_label', 'activityType', 'badgeType', 'min_goal', 'max_goal'],inplace=True)
df.drop(columns=['filteredDemographicVO2Max', 'exertion_points_percentage', 'responsiveness_points_percentage', 'distance', 'scl_avg', 'sleep_duration'], inplace=True)
df.drop(columns=['ENTERTAINMENT', 'GYM', 'HOME', 'HOME_OFFICE', 'OTHER', 'OUTDOORS', 'TRANSIT', 'WORK/SCHOOL'], inplace=True)

################################################################################
# Apliquem la correcció del bmi
################################################################################
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


##################################################################################
# Correcció dels valors anòmals trobats
##################################################################################
# Corregim el dataset.
# Definim els rangs possibles
rangs_possibles = {
    'nightly_temperature':       {'min': 30,    'max': 36},
    'nremhr':                    {'min': 40,    'max': 100},
    'rmssd':                     {'min': 1,     'max': 200},
    'spo2':                      {'min': 95,     'max': 100},
    'full_sleep_breathing_rate': {'min': 5,     'max': 27},
    'stress_score':              {'min': 1,     'max': 100},
    'sleep_points_percentage':   {'min': 0.01,  'max': 1},    # percentatge    
    'daily_temperature_variation':{'min': -6,     'max': 2},
    'calories':                  {'min': 1000,  'max': 6000},    # cal/dia, més de 6000 hauria de ser un error
    'sedentary_minutes':         {'min': 1,     'max': 1200},    # 1200 son 20h de sedentarisme, descartem els errors de 24 h
    'bpm':                       {'min': 40,     'max': 200},
    'lightly_active_minutes':    {'min':1,      'max': 550},
    'minutesAsleep':             {'min': 200,   'max': 800},   # mes de 800 i menys de 200 no hauria de ser co,u
    'minutesAwake':              {'min': 1,     'max': 170},
    'sleep_efficiency':          {'min': 65,    'max': 100},  # S'hauria de recalcular la sleep eficiency
    'sleep_deep_ratio':          {'min': 0.01,  'max': 1},       # ratio [0–1]
    'sleep_wake_ratio':          {'min': 0.01,  'max': 1},
    'sleep_light_ratio':         {'min': 0.01,  'max': 1},
    'sleep_rem_ratio':           {'min': 0.01,  'max': 1},
    'steps':                     {'min': 100,   'max': 39000}
}
# Neteja per columnes
for col, valor in rangs_possibles.items():
    # valors massa baixos
    if valor['min'] is not None:
        df.loc[df[col] < valor['min'], col] = np.nan
    # valors massa alts
    if valor['max'] is not None:
        df.loc[df[col] > valor['max'], col] = np.nan

##################################################################################
# Fem un drop de altres possibles columnes que ens pugui millorar la predicció del sistema segons els resultats del EDA
##################################################################################
# Targets que no volem predir que no ens pot aportar el fitbit
df.drop(columns=['ALERT', 'HAPPY', 'NEUTRAL', 'SAD', 'RESTED/RELAXED'], inplace=True)
# Eliminem els de sleep 
df.drop(columns=['sleep_points_percentage', 'minutesToFallAsleep', 'minutesAfterWakeup'], inplace=True)
# Eliminem els de activitat i zones hr
df.drop(columns=['minutes_in_default_zone_3', 'minutes_in_default_zone_2', 'minutes_in_default_zone_1', 'minutes_below_default_zone_1'], inplace=True)
df.drop(columns=['very_active_minutes', 'moderately_active_minutes', 'lightly_active_minutes'], inplace=True)

#################################################################
# Exportem les dades preprocessades a csv i comprovem
#################################################################

df.to_csv('data/df_cleaned.csv', index=False)
print(df.info())
 
