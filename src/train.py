# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from src.preprocess import load_data, prepare_Xy, save_preprocessor
from src.models import build_random_forest, save_sklearn_model

def train_main(data_path="data/diabetes_sample.csv", out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(data_path)

    X, y, pre = prepare_Xy(df, training=True)
    save_preprocessor(pre, os.path.join(out_dir, "preprocessor.joblib"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = build_random_forest()
    model.fit(X_train_res, y_train_res)

    save_sklearn_model(model, os.path.join(out_dir, "model.joblib"))

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("AUC:", roc_auc_score(y_test, y_proba))
    print("Recall:", recall_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_main()
