from ai_health_assistant.utils.prep_helpers import preprocess_data, FEATURES, TARGET
import pandas as pd

'''
Preprocessament de les dades:
Aquest script llegeix els conjunts d'entrenament i prova ja netejats,
els preprocessa de forma separada per evitar data leakage, i els guarda
en format preparat per a l'entrenament del model.
'''



df_train, df_test, preprocessor = preprocess_data(
    train_path='data/df_cleaned_train.csv',
    test_path='data/df_cleaned_test.csv',
    output_dir='data/df_engineered',
    features=FEATURES,
    target=TARGET
)

df = pd.concat([df_train, df_test], axis=0)
print(df.info())