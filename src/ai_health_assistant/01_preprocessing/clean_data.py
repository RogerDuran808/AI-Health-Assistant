from ai_health_assistant.utils.clean_helpers import clean_data


########################## Neteja de les dades ################################
'''
Justificació de la neteja de dades al notebook: 01_LifeSnaps_EDA.ipynb
L'objectiu es tenir les dades netes i imputades de forma basica per fer un bon preprocessament

Seleccionem les columnes disponibles del dispositiu fitbit
Corregim la variable del bmi i afegim categories de bmi_tipo
Corregim els valors anòmals del dataset
Finalment exportem a CSV
'''

FEATURES = [
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

df_train, df_test, df = clean_data(
    input_path='data/daily_fitbit_sema_df_unprocessed.csv', 
    output_path='data/df_cleaned',
    target='TIRED',
    features=FEATURES
    )

print(df.info())
