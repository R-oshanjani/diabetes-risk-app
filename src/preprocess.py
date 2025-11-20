# src/preprocess.py
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

NUM_FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessor():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_FEATURES)
    ])
    return preprocessor

def prepare_Xy(df, preprocessor=None, training=True):
    X = df[NUM_FEATURES].copy()
    y = df['Outcome'].copy()
    if training:
        preprocessor = preprocessor or build_preprocessor()
        Xt = preprocessor.fit_transform(X)
    else:
        Xt = preprocessor.transform(X)
    return Xt, y, preprocessor

def save_preprocessor(preprocessor, path='models/preprocessor.joblib'):
    joblib.dump(preprocessor, path)
