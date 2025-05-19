from ai_health_assistant.utils.clean_helpers import clean_data


########################## Neteja de les dades ################################
'''
Justificació de la neteja de dades al notebook: 01_LifeSnaps_EDA.ipynb
L'objectiu es tenir les dades netes i imputades de forma basica per fer un bon preprocessament

Eliminem les columnes irrellevants per la nostre predicció.
Corregim la variable del bmi i afegim categories de bmi_tipo
Corregim els valors anòmals del dataset
Fem un drop de columnes segons el EDA realitzat que creiem que no aporten valor a la predicció
Finalment exportem a CSV
'''


df_clean = clean_data(
    input_path='data/daily_fitbit_sema_df_unprocessed.csv', 
    output_path='data/df_cleaned.csv'
    )

print(df_clean.info())
