# Dades
import pandas as pd
import numpy as np

# Preprocessament de dades
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

####################

# Llegim el fitxer CSV ja netejat previament a "data/cleaned_data.py"
df = pd.read_csv('data/df_cleaned.csv')
print("Data importada correctament")

#################################
# Tractament de valors Absents
################################
# Imputació de dades del training set


# Definim X, y
y_tense = df['TENSE/ANXIOUS']
X = df.drop(columns=['TENSE/ANXIOUS', 'RESTED/RELAXED'])

numeric_X = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categoric_X = X.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()

######################

# Transformem les columnes amb pipeline
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])

categoric_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_X),
    ("cat", categoric_pipe, categoric_X)
])

###############



