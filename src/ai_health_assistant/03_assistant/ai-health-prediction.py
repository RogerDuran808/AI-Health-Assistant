import pandas as pd
import joblib

# Carreguem el dataset ja pre‐processat
df = pd.read_csv('data/df_engineered_test.csv')

# Extreu una fila de prova
prova = df.sample(n=1, random_state=40)
idx = prova.index[0]
print(f"Mostrant la fila amb índex {idx}:\n", prova, "\n")

# Separa característiques (X) i elimna les columnes de target
X_test = prova.drop(columns=['TIRED'])

# Carrega els models serialitzats
model = joblib.load('models/LGBM_TIRED.joblib')

# Predicció per a cada etiqueta
pred      = model.predict(X_test)[0]
prob_pos  = model.predict_proba(X_test)[:,1][0]


# Resultats
print(f"TIRED → pred: {pred}, prob(1): {prob_pos:.3f}")
