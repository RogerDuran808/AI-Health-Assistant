from ai_health_assistant.utils.clean_helpers import clean_data, TARGET, FEATURES
import pandas as pd

########################## Neteja de les dades ################################
'''
Justificació de la neteja de dades al notebook: 01_LifeSnaps_EDA.ipynb
L'objectiu es tenir les dades netes i imputades de forma basica per fer un bon preprocessament

Seleccionem les columnes disponibles del dispositiu fitbit
Corregim la variable del bmi i afegim categories de bmi_tipo
Corregim els valors anòmals del dataset
Finalment exportem a CSV
'''


df_train, df_test = clean_data(
    input_path='data/daily_fitbit_sema_df_unprocessed.csv', 
    output_path='data/df_cleaned',
    target=TARGET,
    features=FEATURES
    )

df = pd.concat([df_train, df_test], axis=0)
print(df.info())
