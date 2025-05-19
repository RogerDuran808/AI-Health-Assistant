# Dades
import pandas as pd
import numpy as np

# Preprocessament de dades
from sklearn.preprocessing import StandardScaler,  OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

####################

# Llegim el fitxer CSV ja netejat previament a "data/cleaned_data.py"
df = pd.read_csv('data/df_cleaned.csv')
print("Data importada correctament")

target = ['TIRED']
features = ['steps', 'calories', 'bpm', 'sedentary_minutes', 'resting_hr', 'minutesAsleep', 'bmi_tipo']

df = df[features + target]

df.dropna(subset=target, inplace=True)

#################################
# Tractament de valors Absents
################################
# Imputació de dades del training set
print(df.isnull().mean()*100)

# Definim X, y
y = df['TIRED']
X = df.drop(columns=['TIRED'])


categoric_X = ['bmi_tipo']
numeric_X = X.drop(columns=categoric_X).columns.tolist()

print(f"Numeric columns: {numeric_X}")
print(f"Categoric columns: {categoric_X}")

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("power",   PowerTransformer(method="yeo-johnson", standardize=False)),
    ("scaler", StandardScaler())
])

categoric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[["Infrapes","Normal","Sobrepes","Obes"]]))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_X),
    ("cat", categoric_pipe, categoric_X)
])

X_prep = preprocessor.fit_transform(X)
features = preprocessor.get_feature_names_out()

index_df = df.index

df_prep = pd.concat([pd.DataFrame(X_prep, columns=features, index=index_df), y], axis=1)
print(df_prep.isnull().mean()*100)

df_prep.to_csv('data/df_preprocessed.csv', index=False)
###############



