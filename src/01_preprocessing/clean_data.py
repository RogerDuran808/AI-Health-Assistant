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
# Importa el fitxer CSV
df = pd.read_csv('D:/roger/OneDrive/Documentos/02. Formació/2. UNIVERSITAT/05. TFG/03. Datasets/Lifesnaps Fitbit/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv')

# Eliminem les columnes que no necessitem:
# Motius: 
# - dades que podem obtenir a traves del nostre fitbit inspire 3
# - columnes irrellevants que no aporten valor
# - filtratge de les columnes segons el PCA 95% (amb 17-18 columnes en fem prou)
df.drop(columns=['Unnamed: 0', 'id', 'date', 'mindfulness_session', 'step_goal', 'step_goal_label', 'ENTERTAINMENT', 'GYM', 'HOME', 'HOME_OFFICE', 'OTHER', 'OUTDOORS', 'TRANSIT', 'WORK/SCHOOL', 'activityType', 'badgeType', 'filteredDemographicVO2Max', 'exertion_points_percentage', 'responsiveness_points_percentage', 'distance', 'scl_avg', 'sleep_duration', 'min_goal', 'max_goal'], inplace=True)

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
    'nightly_temperature':       {'min': 30,    'max': None},
    'nremhr':                    {'min': 30,    'max': 100},
    'rmssd':                     {'min': 1,     'max': 200}, 
    'full_sleep_breathing_rate': {'min': 1,     'max': None},
    'stress_score':              {'min': 1,     'max': 100},
    'sleep_points_percentage':   {'min': 0.01,  'max': 1},    # percentatge
    'calories':                  {'min': 1000,  'max': 6000},    # cal/dia, més de 6000 hauria de ser un error
    'sedentary_minutes':         {'min': 1,     'max': 1200},    # 1200 son 20h de sedentarisme, descartem els errors de 24 h
    'lightly_active_minutes':    {'min':1,      'max': None},
    'minutesAsleep':             {'min': 1,     'max': 14*60},   # màx 14h, més crec que es tractaria d'un error
    'minutesAwake':              {'min': 0,     'max': 200},
    'sleep_efficiency':          {'min': 65,    'max': 100},
    'sleep_deep_ratio':          {'min': 0.01,  'max': 1},       # ratio [0–1]
    'sleep_wake_ratio':          {'min': 0.01,  'max': 1},
    'sleep_light_ratio':         {'min': 0.01,  'max': 1},
    'sleep_rem_ratio':           {'min': 0.01,  'max': 1},
    'steps':                     {'min': 100,   'max': 35000},
}
# Neteja per columnes
for col, b in rangs_possibles.items():
    # valors massa baixos
    if b['min'] is not None:
        df.loc[df[col] < b['min'], col] = np.nan
    # valors massa alts
    if b['max'] is not None:
        df.loc[df[col] > b['max'], col] = np.nan


#################################################################
# Fem un drop de altres possibles columnes que ens pugui millorar la predicció del sistema segons els resultats del EDA
##################################################################################
'''
Gràcies a l'analisi del PCA, podem observar que aproximadament amb 17-18
columnes podem obtenir el 95% de la informació, com que ara tenim 40 columnes, en 
podriem eliminar fina a 20 sense perdre informació.

Propostes de columnes a eliminar segons EDA:
- Segons estadistica descriptiva:
-- Per poques dades (<2000): spo2, stress score, sleep_points_percentage.

- Segons l'anàlisi d'assimetria i curtosi:
-- variables extremadament asimètriques: minutesToFallAsleep, minutesAfterWakeup, minutes_in_deafault_zone_3, , minutes_in_deafault_zone_2, moderate_active_minutes, very_active_minutes.
-- aquestes variable amb asimetria (skew) sueperior a 1, es podrien eliminar o fer logaritme per atenuar el seu efecte.

- Variables categòriques:
-- Eliminar els 'tagets' que no utilitzarem.: ALERT, HAPPY,...

- Segons correlació (> 0.7)
-- Variables altament correlacionades amb una altre que aporta més info: 
nremhr                        resting_hr           0.848725
moderately_active_minutes     steps                0.716821
minutes_below_default_zone_1  sedentary_minutes   -0.720506
lightly_active_minutes        sedentary_minutes   -0.798791

--- Podriem eliminar:
--- nremhr ja que te menys dades i més asimetria (tot hi que es baixa)
--- moderate_active_minutes, ja que te molta més asimetria que steps,
moderate active mintes te moltes més dades pero podria ser que aquestes dades
fosin de mala qualitat o erronies.
--- provarem de eliminar sedentary minutes per
--- llavors també podriem eliminar lightly_active_minutes.

- 
'''
df.drop(columns=['ALERT', 'HAPPY', 'NEUTRAL', 'SAD', 'RESTED/RELAXED'], inplace=True)
# Eliminem els de sleep 
df.drop(columns=['sleep_points_percentage', 'minutesToFallAsleep', 'minutesAfterWakeup'], inplace=True)
# Eliminem els de activitat i zones hr
df.drop(columns=['minutes_in_default_zone_3', 'minutes_in_default_zone_2', 'minutes_in_default_zone_1', 'minutes_below_default_zone_1'], inplace=True)
df.drop(columns=['very_active_minutes', 'moderately_active_minutes', 'lightly_active_minutes'], inplace=True)
df.drop(columns=['nremhr'], inplace=True)

df.to_csv('data/df_cleaned.csv', index=False)
print(df.info())
