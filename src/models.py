# src/models.py
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def build_random_forest():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

def save_sklearn_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
