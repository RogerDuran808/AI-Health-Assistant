import pandas as pd

# ----------------------------------------------------------------------------------
# 1. Carrega del conjunt de dades
# ----------------------------------------------------------------------------------
DATA_PATH = "data/df_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# ----------------------------------------------------------------------------------
# 2. Funcions d’utilitat
# ----------------------------------------------------------------------------------
def top10(df_subset: pd.DataFrame, col: str, ascending: bool) -> pd.DataFrame:
    """
    Retorna el Top-10 (o pitjor-10) per a una columna donada,
    després d’aplicar els filtres previs de 'df_subset'.
    """
    return df_subset.sort_values(by=col, ascending=ascending).head(10)

# ----------------------------------------------------------------------------------
# 3. Rànquings "pitjors" per mètrica
# ----------------------------------------------------------------------------------

# 3·1 Resting Heart Rate – pitjor = valor més ALT
worst_resting_hr = top10(df, "resting_hr", ascending=False)

# 3·2 RMSSD – pitjor = valor més BAIX (però >0 i <200 ms)
rmssd_feasible = df[(df["rmssd"] > 0) & (df["rmssd"] < 200)]
worst_rmssd = top10(rmssd_feasible, "rmssd", ascending=True)

# 3·3 SpO₂ – pitjor = valor més BAIX (70 % ≤ SpO₂ ≤ 100 %)
spo2_feasible = df[df["spo2"].between(70, 100, inclusive="both")]
worst_spo2 = top10(spo2_feasible, "spo2", ascending=True)

# 3·4 Variació temperatura cutània – pitjor = desviació més ALTA
#     (rang factible fixat entre −5 °C i +5 °C)
temp_feasible = df[df["daily_temperature_variation"].between(-5, 5, inclusive="both")]
worst_temp = top10(temp_feasible, "daily_temperature_variation", ascending=False)

# 3·5 Freqüència respiratòria nocturna – pitjor = valor més ALT (5–40 rpm)
breath_feasible = df[df["full_sleep_breathing_rate"].between(5, 40, inclusive="both")]
worst_breath = top10(breath_feasible, "full_sleep_breathing_rate", ascending=False)

# ----------------------------------------------------------------------------------
# 4. Sortida (print o guarda)
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
worst_resting_hr.to_csv("data/data_llm/top10_resting_hr.csv", index=False)
worst_rmssd.to_csv("data/data_llm/top10_rmssd.csv", index=False)
worst_spo2.to_csv("data/data_llm/top10_spo2.csv", index=False)
worst_temp.to_csv("data/data_llm/top10_temp.csv", index=False)
worst_breath.to_csv("data/data_llm/top10_breath.csv", index=False)


