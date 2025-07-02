from ai_health_assistant.utils.clean_helpers import clean_data, TARGET, FEATURES
import pandas as pd

########################## Neteja de les dades ################################
'''
Exploració de dades inicial realitzada a l'arxiu: 01_LifeSnaps_EDA.ipynb
L'objectiu és tenir les dades netes i imputades de forma bàsica per fer un bon preprocessament. 
Sense errors de dades i amb tractament d'outliers per a millorar la qualitat del model.

Seleccionem les columnes disponibles del dispositiu Fitbit Inspire 3
Corregim la variable del bmi i afegim categories de bmi_tipo
Corregim els valors anòmals del dataset
Finalment exportem a CSV, les dades separades del train i del test
'''


df_train, df_test = clean_data(
    input_path='data/daily_fitbit_sema_df_unprocessed.csv', 
    output_path='data/df_cleaned',
    target=TARGET,
    features=FEATURES
    )

df = pd.concat([df_train, df_test], axis=0)
df.to_csv('data/df_cleaned.csv', index=False)
print(df.info())
