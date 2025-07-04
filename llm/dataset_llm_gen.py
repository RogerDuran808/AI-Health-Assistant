import pandas as pd

# ----------------------------------------------------------------------------------
# Carrega del conjunt de dades
# ----------------------------------------------------------------------------------
DATA_PATH = "data/df_cleaned.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna()
# ----------------------------------------------------------------------------------
# Funcions d’utilitat
# ----------------------------------------------------------------------------------
def top10(df_subset: pd.DataFrame, col: str, ascending: bool) -> pd.DataFrame:
    """
    Retorna el Top-10 (o pitjor-10) per a una columna donada,
    després d’aplicar els filtres previs de 'df_subset'.
    """
    return df_subset.sort_values(by=col, ascending=ascending).head(10)
# Mètriques d'interès
METRIQUES = [
    "resting_hr",                    # pulsacions per minut en repòs
    "rmssd",                        # variabilitat de la FC (ms)
    "spo2",                         # saturació d'oxigen (%)
    "daily_temperature_variation",  # variació de Tª cutània (°C vs. basal)
    "full_sleep_breathing_rate"      # respiracions/min durant el son
]
# ----------------------------------------------------------------------------------
# Rànquings "pitjors" per mètrica
# ----------------------------------------------------------------------------------

# Resting Heart Rate – pitjor = valor més ALT
worst_resting_hr = top10(df, "resting_hr", ascending=False)

# RMSSD – pitjor = valor més BAIX (però >0 i <200 ms)
rmssd_feasible = df[(df["rmssd"] > 0) & (df["rmssd"] < 200)]
worst_rmssd = top10(rmssd_feasible, "rmssd", ascending=True)

# SpO₂ – pitjor = valor més BAIX (70 % ≤ SpO₂ ≤ 100 %)
spo2_feasible = df[df["spo2"].between(70, 100, inclusive="both")]
worst_spo2 = top10(spo2_feasible, "spo2", ascending=True)

# Variació temperatura cutània – pitjor = desviació més ALTA
#     (rang factible fixat entre −5 °C i +5 °C)
temp_feasible = df[df["daily_temperature_variation"].between(-5, 5, inclusive="both")]
worst_temp = top10(temp_feasible, "daily_temperature_variation", ascending=False)

# Freqüència respiratòria nocturna – pitjor = valor més ALT (5–40 rpm)
breath_feasible = df[df["full_sleep_breathing_rate"].between(5, 40, inclusive="both")]
worst_breath = top10(breath_feasible, "full_sleep_breathing_rate", ascending=False)

# ----------------------------------------------------------------------------------
# Sortida (print o guarda)
# ----------------------------------------------------------------------------------
print("== Top-10 Resting HR (alt) ==")
print(worst_resting_hr, end="\n\n")

print("== Top-10 RMSSD (baix) ==")
print(worst_rmssd, end="\n\n")

print("== Top-10 SpO₂ (baix) ==")
print(worst_spo2, end="\n\n")

print("== Top-10 ΔSkinTemp (alt) ==")
print(worst_temp, end="\n\n")

print("== Top-10 Sleep Breathing Rate (alt) ==")
print(worst_breath, end="\n\n")

# Exportem:
worst_resting_hr[METRIQUES].to_csv("data/data_llm/top10_resting_hr.csv", index=False)
worst_rmssd[METRIQUES].to_csv("data/data_llm/top10_rmssd.csv", index=False)
worst_spo2[METRIQUES].to_csv("data/data_llm/top10_spo2.csv", index=False)
worst_temp[METRIQUES].to_csv("data/data_llm/top10_temp.csv", index=False)
worst_breath[METRIQUES].to_csv("data/data_llm/top10_breath.csv", index=False)




# Obtenció de les 50 millors files completament sanes

# -------------------------------------------------------------------
# Definir els rangs “completament sans”
# -------------------------------------------------------------------
cond_resting_hr   = df["resting_hr"].between(40, 70, inclusive="both")
cond_rmssd        = df["rmssd"].between(30, 200, inclusive="both")
cond_spo2         = df["spo2"].between(95, 100, inclusive="both")
cond_temp_var     = df["daily_temperature_variation"].between(-0.5, 0.5, inclusive="both")
cond_breath_rate  = df["full_sleep_breathing_rate"].between(12, 20, inclusive="both")

healthy_mask = cond_resting_hr & cond_rmssd & cond_spo2 & cond_temp_var & cond_breath_rate
healthy = df[healthy_mask].copy()

## -------------------------------------------------------------------
# Columnes auxiliars per mesurar “proximitat” als valors ideals
# -------------------------------------------------------------------
df_aux = (
    healthy.assign(
        abs_temp   = healthy["daily_temperature_variation"].abs(),           # |ΔT|
        breath_dev = (healthy["full_sleep_breathing_rate"] - 14).abs()       # |resp − 14 rpm|
    )
)

# -------------------------------------------------------------------
# Classificació: HRV ↑, SpO₂ ↑, |ΔT| ↓, HR ↓, |resp−14| ↓
# -------------------------------------------------------------------
df_sorted = (
    df_aux
    .sort_values(
        by=["rmssd", "spo2", "abs_temp", "resting_hr", "breath_dev"],
        ascending=[False, False, True, True, True]
    )
    .drop(columns=["abs_temp", "breath_dev"])           # netegem auxiliars
)

# -------------------------------------------------------------------
# Seleccionar les 50 files més saludables
# -------------------------------------------------------------------
top50_healthy = df_sorted[METRIQUES].head(50).reset_index(drop=True)

print(top50_healthy)

# Opcional: guardar resultat
top50_healthy.to_csv("data/data_llm/top50_healthy_samples.csv", index=False)