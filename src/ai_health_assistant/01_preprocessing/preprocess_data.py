from ai_health_assistant.utils.prep_helpers import preprocess_data
from pathlib import Path

'''
Preprocessament de les dades:
Aquest script llegeix els conjunts d'entrenament i prova ja netejats,
els preprocessa de forma separada per evitar data leakage, i els guarda
en format preparat per a l'entrenament del model.
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


X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    train_path='data/df_cleaned_train.csv',
    test_path='data/df_cleaned_test.csv',
    output_dir='data/preprocessed',
    target='TIRED',
    features=FEATURES
)
    
print(f"- Mostres d'entrenament: {len(X_train)}")
print(f"- Mostres de prova: {len(X_test)}")
print(f"- Característiques: {X_train.shape[1]}")