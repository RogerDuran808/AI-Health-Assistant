from ai_health_assistant.utils.prep_helpers import preprocess_data

'''
Preprocessament de les dades:
Un cop netejades les dades, les preprocessem, per preparar-les per l'entrenament del model.
'''

df_prep = preprocess_data(
    input_path="data/df_cleaned.csv",
    output_path="data/df_preprocessed.csv"
)
print(df_prep.info())


