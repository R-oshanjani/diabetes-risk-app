# src/evaluate.py
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shap_summary(model_path='models/model.joblib', preproc_path='models/preprocessor.joblib', sample_csv='data/diabetes_sample.csv'):
    model = joblib.load(model_path)
    pre = joblib.load(preproc_path)
    df = pd.read_csv(sample_csv)
    X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    Xt = pre.transform(X)

    # For tree-based model:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt)
        shap.summary_plot(shap_values, Xt, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        print("Saved shap_summary.png")
    except Exception as e:
        print("SHAP TreeExplainer failed:", e)
