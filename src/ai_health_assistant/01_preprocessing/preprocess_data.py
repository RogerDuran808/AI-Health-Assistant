from ai_health_assistant.utils.prep_helpers import preprocess_data

'''
Preprocessament de les dades:
Un cop netejades les dades, les preprocessem, per preparar-les per l'entrenament del model.
Les funcions de preprocessament son les definides als prep_helpers.py

'''
# Target a predir
TARGET = "TIRED"

# Features que volem utilitzar per fer la predicció
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
    'wake_after_sleep_pct'
] 


df_prep = preprocess_data(
    input_path="data/df_cleaned.csv",
    output_path="data/df_preprocessed.csv",
    target=TARGET,
    features=FEATURES)

print(df_prep.info())


