from ai_health_assistant.utils.prep_helpers import preprocess_data, FEATURES, TARGET, COLUMNES_DATASET
import pandas as pd

'''
Preprocessament de les dades:
Aquest script llegeix els conjunts d'entrenament i prova ja netejats,
aplica feature engineering a cada conjunt per separat, i els guarda
en format preparat per a l'entrenament del model i el preprocessament del model
realitzat en el fitxer training.py.
'''



df_train, df_test = preprocess_data(
    train_path='data/df_cleaned_train.csv',
    test_path='data/df_cleaned_test.csv',
    output_dir='data/df_engineered',
    features=COLUMNES_DATASET,
    target=TARGET
)

df = pd.concat([df_train, df_test], axis=0)
print(df.info())