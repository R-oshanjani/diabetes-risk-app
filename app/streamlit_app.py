# app/streamlit_app.py
import os
from pathlib import Path
import json
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt

# Try to import shap; gracefully handle if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -----------------------
# Paths (robust relative)
# -----------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
LOCAL_MODEL_PATH = MODELS_DIR / "model.joblib"
LOCAL_PRE_PATH = MODELS_DIR / "preprocessor.joblib"

# Backend URL (if you run the FastAPI backend)
BACKEND_URL = "http://127.0.0.1:8000"

# SQLite DB for prediction history (in repo root)
DB_PATH = REPO_ROOT / "predictions.db"

# Feature order expected by preprocessor
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# -----------------------
# Helpers
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS preds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            prob REAL,
            pred INTEGER,
            payload TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def save_prediction(prob, pred, payload):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO preds(ts, prob, pred, payload) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), float(prob), int(pred), json.dumps(payload)),
    )
    conn.commit()
    conn.close()

def load_history(limit=100):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT id, ts, prob, pred, payload FROM preds ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    if not df.empty:
        df["payload"] = df["payload"].apply(lambda s: json.dumps(json.loads(s), ensure_ascii=False))
    return df

@st.cache_resource
def load_local_artifacts():
    """Load model and preprocessor if present; return (model, pre) or (None, None)."""
    try:
        if LOCAL_MODEL_PATH.exists() and LOCAL_PRE_PATH.exists():
            model = joblib.load(LOCAL_MODEL_PATH)
            pre = joblib.load(LOCAL_PRE_PATH)
            return model, pre
    except Exception as e:
        st.error(f"Failed to load local artifacts: {e}")
    return None, None

def call_backend_predict(payload):
    """Call backend /predict endpoint. Returns dict or raises Exception."""
    url = f"{BACKEND_URL.rstrip('/')}/predict"
    resp = requests.post(url, json=payload, timeout=3)
    resp.raise_for_status()
    return resp.json()

def predict_local(model, pre, payload_df):
    """Return probability (float) and pred (0/1)."""
    Xt = pre.transform(payload_df[FEATURE_NAMES])
    proba = float(model.predict_proba(Xt)[0, 1])
    pred = int(model.predict(Xt)[0])
    return proba, pred, Xt

def safe_shap_summary_plot(model, Xt, feature_names):
    """Create a shap summary plot to a figure and return the figure."""
    # SHAP uses the global figure internally; we create a figure and call summary_plot with show=False
    fig = plt.figure(figsize=(6, 3))
    try:
        # For tree models: TreeExplainer is faster/stable
        if hasattr(model, "feature_importances_") and SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xt)
            # shap_values may be list for multiclass; we handle binary classifier default
            shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False)
        elif SHAP_AVAILABLE:
            explainer = shap.Explainer(model.predict_proba, Xt)
            shap_values = explainer(Xt)
            shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False)
        else:
            raise RuntimeError("SHAP not available")
    except Exception:
        plt.close(fig)
        raise
    return fig

def fallback_feature_importance_figure(model, feature_names):
    """Return a matplotlib fig with barh feature importance (if available)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importances, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        ax.set_xlabel("Feature importance")
        ax.set_title("Model feature importances")
    else:
        ax.text(0.5, 0.5, "No feature importances available", ha="center", va="center")
        ax.axis("off")
    return fig

# -----------------------
# App UI
# -----------------------
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("Diabetes Risk Predictor — Dashboard")
st.markdown("**Demo**: enter diagnostic values, get a risk probability, and inspect model explanations. Not medical advice.")

# Load artifacts
local_model, local_pre = load_local_artifacts()

# Sidebar with quick info & sample presets
with st.sidebar:
    st.header("Quick controls")
    st.write("Model status:")
    if local_model is None:
        st.write("• Local model: **not found**")
    else:
        st.write("• Local model: **loaded**")
    st.write(f"• Backend URL: `{BACKEND_URL}`")
    st.markdown("---")
    st.subheader("Sample patients")
    sample_choice = st.selectbox(
        "Pick a demo patient",
        options=[
            "Custom (empty)",
            "Healthy young (low risk)",
            "Middle age (moderate risk)",
            "Older high BMI (higher risk)",
        ],
    )

# Prepare default values depending on sample
if sample_choice == "Healthy young (low risk)":
    defaults = {"Pregnancies": 0, "Glucose": 90.0, "BloodPressure": 70.0, "SkinThickness": 15.0, "Insulin": 50.0, "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 25}
elif sample_choice == "Middle age (moderate risk)":
    defaults = {"Pregnancies": 1, "Glucose": 120.0, "BloodPressure": 75.0, "SkinThickness": 20.0, "Insulin": 80.0, "BMI": 28.0, "DiabetesPedigreeFunction": 0.5, "Age": 45}
elif sample_choice == "Older high BMI (higher risk)":
    defaults = {"Pregnancies": 2, "Glucose": 150.0, "BloodPressure": 80.0, "SkinThickness": 25.0, "Insulin": 140.0, "BMI": 36.0, "DiabetesPedigreeFunction": 1.2, "Age": 60}
else:
    defaults = {k: (0 if k=="Pregnancies" else 120.0 if k=="Glucose" else 70.0 if k=="BloodPressure" else 20.0 if k=="SkinThickness" else 80.0 if k=="Insulin" else 32.0 if k=="BMI" else 0.5 if k=="DiabetesPedigreeFunction" else 33) for k in FEATURE_NAMES}

# Input form
with st.form("predict_form"):
    cols = st.columns(2)
    inputs = {}
    for i, feat in enumerate(FEATURE_NAMES):
        col = cols[i % 2]
        default_val = defaults.get(feat, 0)
        if feat in ["Pregnancies", "Age"]:
            inputs[feat] = col.number_input(feat, min_value=0, max_value=120, value=int(default_val))
        elif feat == "DiabetesPedigreeFunction":
            inputs[feat] = col.number_input(feat, min_value=0.0, max_value=10.0, value=float(default_val), format="%.3f")
        else:
            inputs[feat] = col.number_input(feat, min_value=0.0, max_value=1000.0, value=float(default_val), format="%.3f")
    submitted = st.form_submit_button("Predict")

# Initialize DB
init_db()

# Prediction flow
if submitted:
    payload = {k: float(v) for k, v in inputs.items()}
    # Try backend first
    backend_result = None
    try:
        backend_result = call_backend_predict(payload)
    except Exception:
        backend_result = None

    if backend_result:
        prob = float(backend_result.get("probability", backend_result.get("prob", 0.0)))
        pred = int(backend_result.get("prediction", backend_result.get("pred", 0)))
        st.success(f"Backend prediction: probability={prob:.3f}, class={'High' if pred==1 else 'Low'}")
        save_prediction(prob, pred, payload)
    else:
        # Use local model
        if local_model is None or local_pre is None:
            st.error("No backend available and no local model artifacts found. Run training: `python -m src.train`.")
        else:
            df_in = pd.DataFrame([payload], columns=FEATURE_NAMES)
            try:
                prob, pred, Xt = predict_local(local_model, local_pre, df_in)
            except Exception as e:
                st.error(f"Local prediction failed: {e}")
                prob, pred, Xt = None, None, None

            if prob is not None:
                # Big metric
                pct_text = f"{prob*100:.1f}%"
                color_emoji = "🔴" if prob >= 0.5 else "🟡" if prob >= 0.2 else "🟢"
                st.metric(label=f"Diabetes risk probability {color_emoji}", value=pct_text)
                st.write("Predicted class:", "**High risk**" if pred==1 else "Low risk")
                save_prediction(prob, pred, payload)

                # Explanation: SHAP if available else fallback
                st.markdown("### Model explanation")
                if SHAP_AVAILABLE:
                    try:
                        fig_shap = safe_shap_summary_plot(local_model, Xt, feature_names=FEATURE_NAMES)
                        st.pyplot(fig_shap)
                        plt.close(fig_shap)
                    except Exception as e:
                        st.warning(f"SHAP plotting failed: {e} — showing fallback importances.")
                        fig_imp = fallback_feature_importance_figure(local_model, FEATURE_NAMES)
                        st.pyplot(fig_imp)
                        plt.close(fig_imp)
                else:
                    fig_imp = fallback_feature_importance_figure(local_model, FEATURE_NAMES)
                    st.pyplot(fig_imp)
                    plt.close(fig_imp)

# Show prediction history
st.markdown("---")
st.subheader("Prediction history (latest 50)")
hist = load_history(limit=50)
if hist.empty:
    st.info("No predictions yet. Make a prediction to populate history.")
else:
    st.dataframe(hist, use_container_width=True)
    csv = hist.to_csv(index=False)
    st.download_button("Download history CSV", csv, file_name="prediction_history.csv", mime="text/csv")
