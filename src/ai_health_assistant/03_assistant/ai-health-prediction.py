import pandas as pd
import joblib
import pandas as pd

# Dades com a diccionari (columna: valor)
data = {
    "bmi": 25,
    "weight": 80,
    "height": 180,
    "calories": 2663,
    "steps": 9135,
    "lightly_active_minutes": 161,
    "moderately_active_minutes": 3,
    "very_active_minutes": 55,
    "sedentary_minutes": 835,
    "resting_hr": 51.61,
    "minutes_below_default_zone_1": 1347,
    "minutes_in_default_zone_1": 42,
    "minutes_in_default_zone_2": 13,
    "minutes_in_default_zone_3": 0,
    "minutesToFallAsleep": 0,
    "minutesAsleep": 330,
    "minutesAwake": 54,
    "minutesAfterWakeup": 0,
    "sleep_efficiency": 87,
    "sleep_deep_ratio": 0.92,
    "sleep_light_ratio": 0.753,
    "sleep_rem_ratio": 0.47,
    "sleep_wake_ratio": 0.84,
    "daily_temperature_variation": -1.33,
    "rmssd": 52.89,
    "spo2": 94.7,
    "full_sleep_breathing_rate": 14.4,
    "TIRED": 0
}

# Crear el DataFrame amb una sola fila
prova = pd.DataFrame([data])



FEATURES = [
    "bmi",
    "recovery_factor",
    "minutesAsleep",
    "full_sleep_breathing_rate",
    "daily_temperature_variation",
    "minutes_in_default_zone_1",
    "wake_after_sleep_pct",
    "calories",
    "active_to_rest_transition",
    "rmssd",
    "sleep_activity_balance",
    "deep_sleep_score",
    "steps_norm_cal"
]


prova["activity_intensity"] = prova["very_active_minutes"] * 3 + prova["moderately_active_minutes"] * 2 + prova["lightly_active_minutes"]
prova["recovery_factor"] = prova["minutesAsleep"] / (prova["activity_intensity"] + 1e-3)
prova["wake_after_sleep_pct"] = prova["minutesAwake"] / (prova["minutesAwake"] + prova["minutesAsleep"] + 1e-3)
prova["active_to_rest_transition"] = prova["activity_intensity"] / (prova["minutesAsleep"] + prova["minutesAwake"] + 1e-3)
prova["sleep_activity_balance"] = prova["minutesAsleep"] / (prova["very_active_minutes"] + prova["moderately_active_minutes"] + 1e-3)
prova["deep_sleep_score"] = prova["sleep_deep_ratio"] * prova["sleep_efficiency"]
prova["steps_norm_cal"] = prova["steps"] / (prova["calories"] + 1e-3)

print(prova)

# Separa característiques (X) i elimna les columnes de target
PROVA = prova.drop(columns=['TIRED'])
PROVA = prova[FEATURES]



# Carrega els models serialitzats
model = joblib.load('models/LGBM_model.joblib')

# Predicció per a cada etiqueta
pred      = model.predict(PROVA)[0]
prob_pos  = model.predict_proba(PROVA)[:,1][0]


# Resultats
print(f"TIRED → pred: {pred}, prob(1): {prob_pos:.3f}")
