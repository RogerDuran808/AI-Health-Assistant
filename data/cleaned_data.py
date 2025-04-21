# Import de llibreries necessàries
import pandas as pd
import numpy as np

# Importa el fitxer CSV
df = pd.read_csv('D:/roger/OneDrive/Documentos/02. Formació/2. UNIVERSITAT/05. TFG/03. Datasets/Lifesnaps Fitbit/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv')

# Eliminem les columnes que no necessitem
df.drop(columns=['Unnamed: 0', 'id','mindfulness_session', 'step_goal', 'step_goal_label', 'ALERT', 'HAPPY', 'NEUTRAL', 'SAD','TIRED', 'ENTERTAINMENT', 'GYM', 'HOME', 'HOME_OFFICE', 'OTHER', 'OUTDOORS', 'TRANSIT', 'WORK/SCHOOL', 'activityType', 'badgeType', 'filteredDemographicVO2Max', 'exertion_points_percentage', 'responsiveness_points_percentage', 'distance', 'scl_avg', 'sleep_duration', 'min_goal', 'max_goal'], inplace=True)

# Eliminem totes les files que tinguin valors nuls del target 'TENSE/ANXIOUS' i 'RESTED/RELAXED'
df = df.dropna(subset=['TENSE/ANXIOUS','RESTED/RELAXED'])

df.to_csv('data/df_cleaned.csv', index=False)
print(df.info())
