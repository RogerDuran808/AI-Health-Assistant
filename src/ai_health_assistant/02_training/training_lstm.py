# -*- coding: utf-8 -*-
"""Entrenament d'un model LSTM senzill per predir la fatiga del dia següent.
Utilitza finestres temporals de 7 dies (entre 5 i 7) de dades del wearable
per predir l'estat de cansament un dia després.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from ai_health_assistant.utils.prep_helpers import FEATURES, TARGET

WINDOW_SIZE = 7  # longitud de la finestra de dades


def carrega_dades(path: str) -> pd.DataFrame:
    """Carrega i prepara el dataset."""
    df = pd.read_csv(path)
    # Convertim la columna de gènere a valors numèrics
    df["gender"] = df["gender"].map({"MALE": 0, "FEMALE": 1})
    # Omplim possibles valors buits
    df = df.fillna(0)
    # Ens assegurem que l'ordre sigui cronològic
    df = df.sort_index()
    return df


def crea_sequences(df: pd.DataFrame, window: int) -> tuple:
    """Genera les seqüències d'entrada i les etiquetes."""
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df[FEATURES].iloc[i : i + window].values)
        y.append(df[TARGET].iloc[i + window])
    return np.array(X), np.array(y)


# Carreguem les dades d'entrenament i test
train_df = carrega_dades("data/df_engineered_train.csv")
test_df = carrega_dades("data/df_engineered_test.csv")

# Escalem les característiques per millorar l'entrenament
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

# Construim les seqüències de 7 dies
X_train, y_train = crea_sequences(train_df, WINDOW_SIZE)
X_test, y_test = crea_sequences(test_df, WINDOW_SIZE)

# Definim un model LSTM molt bàsic
model = Sequential(
    [
        LSTM(16, input_shape=(WINDOW_SIZE, len(FEATURES))),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Early stopping per evitar sobreentrenament
callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

# Entrenem el model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)

# Avaluem al conjunt de test
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisio de test: {acc:.4f}")

# Guardem el model i l'escalador
model.save("models/lstm_next_day.h5")
joblib.dump(scaler, "models/lstm_next_day_scaler.pkl")
