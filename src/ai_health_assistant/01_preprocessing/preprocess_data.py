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
    'nightly_temperature', 
    'nremhr', 
    'rmssd', 
    'spo2', 
    'full_sleep_breathing_rate', 
    'stress_score', 
    'daily_temperature_variation', 
    'calories', 
    'bpm', 
    'sedentary_minutes',
    'lightly_active_minutes',
    'very_active_minutes',
    'resting_hr', 
    'minutesAsleep', 
    'minutesAwake', 
    'sleep_efficiency', 
    'sleep_deep_ratio', 
    'sleep_wake_ratio', 
    'sleep_light_ratio', 
    'sleep_rem_ratio', 
    'steps', 
    'bmi', 
    'age', 
    'gender', 
    'bmi_tipo',
    'wake_after_sleep_pct'
    ] 


df_prep = preprocess_data(
    input_path="data/df_cleaned.csv",
    output_path="data/df_preprocessed.csv",
    target=TARGET,
    features=FEATURES)

print(df_prep.info())


