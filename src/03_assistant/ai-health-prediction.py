import pandas as pd
import joblib

# Carreguem el dataset ja pre‐processat
df = pd.read_csv('data/df_preprocessed.csv')

# 2. Extreu una fila de prova (seed per reproductibilitat)
prova = df.sample(n=1)
idx = prova.index[0]
print(f"Mostrant la fila amb índex {idx}:\n", prova, "\n")

# 3. Separa característiques (X) i elimna les columnes de target
X_test = prova.drop(columns=['TIRED', 'RESTED/RELAXED'])

# 4. Carrega els models serialitzats
model_tense  = joblib.load('models/best_TIRED')
model_rested = joblib.load('models/best_RESTED_RELAXED.joblib')

# 5. Predicció per a cada etiqueta
pred_tense      = model_tense.predict(X_test)[0]
prob_tense_pos  = model_tense.predict_proba(X_test)[:,1][0]

pred_rested     = model_rested.predict(X_test)[0]
prob_rested_pos = model_rested.predict_proba(X_test)[:,1][0]

# 6. Resultats
print(f"TENSE/ANXIOUS → pred: {pred_tense}, prob(1): {prob_tense_pos:.3f}")
print(f"RESTED/RELAXED → pred: {pred_rested}, prob(1): {prob_rested_pos:.3f}")
