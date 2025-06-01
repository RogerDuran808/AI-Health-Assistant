from ai_health_assistant.utils.prep_helpers import preprocess_data

'''
Preprocessament de les dades:
Un cop netejades les dades, les preprocessem, per preparar-les per l'entrenament del model.
Les funcions de preprocessament son les definides als prep_helpers.py

'''   

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
    
    # Feature engineering, noves columnes
    'wake_after_sleep_pct',
    'steps_norm_cal',
    'deep_sleep_score',
    'active_sedentary_ratio',
    'sleep_activity_balance',
    'bmi_hr_interaction',
    'sleep_quality_index',
    'hr_zone_variability',
    'recovery_factor',
    'sleep_eff_rmssd',
    'active_to_rest_transition',
    'active_to_total_ratio'
] 


df_prep = preprocess_data(
    input_path="data/df_cleaned.csv",
    output_path="data/df_preprocessed.csv",
    target="TIRED",
    features=FEATURES)

print(df_prep.info())


