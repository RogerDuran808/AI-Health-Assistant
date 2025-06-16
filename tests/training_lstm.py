"""
Entrenament d'un model LSTM per predir la fatiga del dia següent.
Utilitza finestres temporals de dades del wearable per predir
l'estat de cansament un dia després.

Executar amb Python 3.10:   py -3.10 tests/training_lstm.py
"""
# --- Imports ---
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import random

# --- Seed for reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Asumim que aquests mòduls existeixen
from ai_health_assistant.utils.prep_helpers import build_preprocessor, FEATURES, TARGET

# --- Constants ---
WINDOW_SIZE = 10

def carrega_dades(path: str) -> pd.DataFrame:
    """Carrega i prepara el dataset."""
    print(f"Carregant dades des de: {path}...")
    df = pd.read_csv(path, index_col='date', parse_dates=True)
    
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('missing')

    df = df.sort_values(by=['id', 'date'])
    return df

def crea_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, window: int) -> tuple:
    """
    Genera seqüències per predir l'estat en l'ÚLTIM dia de la finestra.
    """
    X, y = [], []
    user_col = 'id'

    print("Creant seqüències (predicció de l'últim dia)...")
    for _, group in df.groupby(user_col):
        # Necessitem almenys 'window' dies per crear una seqüència completa.
        if len(group) >= window:
            features = group[feature_cols].values
            target = group[target_col].values
            # <<< CANVI CLAU: El bucle ara va un pas més enllà
            for i in range(len(group) - window + 1):
                # La finestra d'entrada són els 7 dies de dades
                X.append(features[i : i + window])
                # La diana és l'estat del dia 7 (índex 'i + window - 1')
                y.append(target[i + window - 1])
    
    print(f"Seqüències creades: {len(X)} mostres.")
    return np.array(X), np.array(y)

def main():
    """Funció principal per executar l'entrenament."""
    train_df = carrega_dades("data/df_engineered_train.csv")
    test_df = carrega_dades("data/df_engineered_test.csv")

    print("Construint i aplicant el preprocesador...")
    preprocessor = build_preprocessor(train_df, FEATURES)

    X_train_transformed = preprocessor.fit_transform(train_df[FEATURES])
    X_test_transformed = preprocessor.transform(test_df[FEATURES])
    
    new_feature_names = list(preprocessor.get_feature_names_out())

    train_processed_df = pd.DataFrame(X_train_transformed, index=train_df.index, columns=new_feature_names)
    test_processed_df = pd.DataFrame(X_test_transformed, index=test_df.index, columns=new_feature_names)

    cols_to_keep = [TARGET, 'id']
    train_final_df = pd.concat([train_processed_df, train_df[cols_to_keep]], axis=1)
    test_final_df = pd.concat([test_processed_df, test_df[cols_to_keep]], axis=1)
    
    train_final_df.dropna(subset=[TARGET], inplace=True)
    test_final_df.dropna(subset=[TARGET], inplace=True)
    train_final_df[TARGET] = train_final_df[TARGET].astype(int)
    test_final_df[TARGET] = test_final_df[TARGET].astype(int)

    X_train_full, y_train_full = crea_sequences(train_final_df, new_feature_names, TARGET, WINDOW_SIZE)
    X_test, y_test = crea_sequences(test_final_df, new_feature_names, TARGET, WINDOW_SIZE)
    
    if X_train_full.shape[0] == 0:
        print("No s'han pogut generar seqüències d'entrenament.")
        return

    # Creem un conjunt de validació per monitoritzar l'entrenament de manera fiable
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Pesos de classe calculats: {class_weight_dict}")

    # Model robust per a una bona base de rendiment
    model = Sequential([
        LSTM(80, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(40, return_sequences=False),
        BatchNormalization(),
        Dropout(0.4),
        Dense(20, activation='relu'),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Callbacks per a un entrenament intel·ligent
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    print("\nIniciant entrenament del model...")
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )

    # --- AVALUACIÓ FINAL AMB EL LLINDAR PER DEFECTE (0.5) ---
    print("\nAvaluant el model base en el conjunt de test...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nLoss de test: {loss:.4f}")
    print(f"Precisió (accuracy) de test: {accuracy:.4f}")

    print("\nInforme de classificació (llindar = 0.5):")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\nMatriu de confusió (llindar = 0.5):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    main()