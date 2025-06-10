# -*- coding: utf-8 -*-
"""
Entrenament d'un model LSTM per predir la fatiga del dia següent.
Utilitza finestres temporals de 7 dies de dades del wearable per predir
l'estat de cansament un dia després.
"""

# --- Imports ---
# 1. Llibreries estàndard
import os

# 2. Llibreries de tercers
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import tensorflow as tf  # <--- MEJORA: Importar TensorFlow explícitamente
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout # <--- MEJORA: Importar Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 3. Mòduls locals
# Asumimos que estos módulos existen y están accesibles
from ai_health_assistant.utils.prep_helpers import build_preprocessor, FEATURES, TARGET

# --- Constants ---
WINDOW_SIZE = 7  # Longitud de la finestra de dades
MODEL_DIR = "models" # <--- MEJORA: Directorio para guardar modelos

def carrega_dades(path: str) -> pd.DataFrame:
    """Carrega i prepara el dataset."""
    print(f"Carregant dades des de: {path}...")
    df = pd.read_csv(path)
    # Convertim la columna de gènere a valors numèrics
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"MALE": 0, "FEMALE": 1})
    
    # Omplim valors buits de manera diferenciada per tipus de dada
    for col in df.select_dtypes(include=np.number).columns:
        # MEJORA SUGERIDA: Usar ffill() puede ser mejor para series temporales
        # df[col] = df[col].fillna(method='ffill').fillna(0)
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('missing')

    # Ens assegurem que l'ordre sigui cronològic per usuari
    # Asumimos que existe una columna 'user_id' y una de fecha como el índice
    if 'user_id' in df.columns:
        df = df.sort_values(by=['user_id', df.index.name or 'date'])
    else:
        df = df.sort_index()

    return df


def crea_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, window: int) -> tuple:
    """
    Genera les seqüències d'entrada i les etiquetes, agrupant per usuari
    per evitar fuga de dades entre usuaris.
    """
    X, y = [], []
    # <--- MEJORA CRÍTICA: Agrupar por usuario para no crear secuencias inválidas
    # Si no hay 'user_id', tratará todo el dataset como un solo usuario.
    user_col = 'user_id' if 'user_id' in df.columns else 'dummy_user'
    if 'user_id' not in df.columns:
        df['dummy_user'] = 0

    print("Creant seqüències per usuari...")
    for _, group in df.groupby(user_col):
        features = group[feature_cols].values
        target = group[target_col].values
        for i in range(len(group) - window):
            X.append(features[i : i + window])
            y.append(target[i + window])
            
    if 'dummy_user' in df.columns:
        df.drop(columns=['dummy_user'], inplace=True)

    print(f"Seqüències creades: {len(X)} mostres.")
    return np.array(X), np.array(y)


def main():
    """Funció principal per executar l'entrenament."""
    # Carreguem les dades d'entrenament i test
    train_df = carrega_dades("data/df_engineered_train.csv")
    test_df = carrega_dades("data/df_engineered_test.csv")

    # Construim i apliquem el preprocesador
    print("Construint i aplicant el preprocesador...")
    preprocessor = build_preprocessor(train_df, FEATURES)

    # Transformem les dades i creem nous DataFrames
    # <--- SOLUCIÓN AL ERROR ---
    X_train_transformed = preprocessor.fit_transform(train_df[FEATURES])
    X_test_transformed = preprocessor.transform(test_df[FEATURES])
    
    # Obtenim els nous noms de les columnes
    try:
        new_feature_names = preprocessor.get_feature_names_out()
    except AttributeError: # Compatibilidad con versiones antiguas de sklearn
        # Esta es una forma más manual si get_feature_names_out no funciona
        new_feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if trans == 'passthrough':
                new_feature_names.extend(cols)
            else:
                try:
                    new_feature_names.extend(trans.get_feature_names_out(cols))
                except AttributeError:
                    new_feature_names.extend([f"{name}_{c}" for c in cols])


    # Creem DataFrames processats
    train_processed_df = pd.DataFrame(X_train_transformed, index=train_df.index, columns=new_feature_names)
    test_processed_df = pd.DataFrame(X_test_transformed, index=test_df.index, columns=new_feature_names)

    # Afegim les columnes que no formen part de FEATURES (com user_id i el target)
    cols_to_keep = [TARGET]
    if 'user_id' in train_df.columns:
        cols_to_keep.append('user_id')
        
    train_final_df = pd.concat([train_processed_df, train_df[cols_to_keep]], axis=1)
    test_final_df = pd.concat([test_processed_df, test_df[cols_to_keep]], axis=1)
    
    # Construim les seqüències de 7 dies
    X_train, y_train = crea_sequences(train_final_df, new_feature_names, TARGET, WINDOW_SIZE)
    X_test, y_test = crea_sequences(test_final_df, new_feature_names, TARGET, WINDOW_SIZE)
    
    # Comprobamos si se han generado datos
    if X_train.shape[0] == 0:
        print("No s'han pogut generar seqüències d'entrenament. Revisa les dades i la mida de la finestra.")
        return

    # Definim un model LSTM
    # <--- MEJORA: Añadimos una capa Dropout para regularización
    model = Sequential([
        LSTM(32, input_shape=(WINDOW_SIZE, len(new_feature_names)), return_sequences=False), # Se puede probar con más neuronas
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Early stopping per evitar sobreentrenament
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    # Entrenem el model
    print("\nIniciant entrenament del model...")
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32, # <--- MEJORA: batch_size de 32 suele ser más eficiente
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Avaluem al conjunt de test
    print("\nAvaluant el model en el conjunt de test...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss de test: {loss:.4f}")
    print(f"Precisió de test: {acc:.4f}")

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\nInforme de classificació:")
    print(classification_report(y_test, y_pred, digits=4))

    # <--- MEJORA: Guardar el modelo y el preprocesador
    print("Guardant el model i el preprocesador...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "fatigue_lstm_model.keras"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    print(f"Model i preprocesador guardats a la carpeta '{MODEL_DIR}'.")


if __name__ == "__main__":
    # Suprimeix logs de TensorFlow menys importants
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    main()