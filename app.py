import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ===============================
# LOAD ARTIFACTS (DEPLOY SAFE)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "pkl")

def load_pkl(filename):
    path = os.path.join(PKL_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

artifact = load_pkl("best_model.pkl")
model = artifact["model"]
THRESHOLD = artifact["threshold"]
MODEL_NAME = artifact.get("name", "BestModel")

scaler = load_pkl("scaler.pkl")
selected_features = load_pkl("selected_features.pkl")
encoding_strategy = load_pkl("encoding_strategy.pkl")
encoders = load_pkl("encoders.pkl")

# ===============================
# HOME ROUTE
# ===============================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_text=None)

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # STEP 1: READ FORM DATA
        input_data = dict(request.form)
        df = pd.DataFrame([input_data])

        # STEP 2: NUMERIC CASTING
        numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # STEP 3: APPLY TRAINED ENCODING
        encoded_values = {}

        for col, method in encoding_strategy.items():
            if col not in df.columns:
                continue

            value = df[col].iloc[0]

            if method == "OneHot":
                ohe = encoders[col]
                arr = ohe.transform(df[[col]])
                ohe_cols = ohe.get_feature_names_out([col])
                for i, c in enumerate(ohe_cols):
                    encoded_values[c] = arr[0][i]

            elif method == "Label":
                le = encoders[col]
                if value not in le.classes_:
                    value = le.classes_[0]
                encoded_values[col] = le.transform([value])[0]

            elif method in ["Frequency", "Count"]:
                mapper = encoders[col]
                encoded_values[col] = mapper.get(value, 0)

            elif method == "TargetMean":
                mapper = encoders[col]["map"]
                fill = encoders[col]["fill"]
                encoded_values[col] = mapper.get(value, fill)

            elif method == "Ordinal":
                oe = encoders[col]
                encoded_values[col] = oe.transform([[value]])[0][0]

            elif method == "Hashing":
                hasher = encoders[col]
                arr = hasher.transform([[str(value)]]).toarray()
                for i in range(arr.shape[1]):
                    encoded_values[f"{col}_hash_{i}"] = arr[0][i]

        # STEP 4: FEATURE ALIGNMENT
        df_final = pd.DataFrame(0.0, index=[0], columns=selected_features)

        for col in selected_features:
            if col in encoded_values:
                df_final[col] = encoded_values[col]
            elif col in df.columns:
                df_final[col] = df[col].iloc[0]

        # STEP 5: SCALING
        X_scaled = scaler.transform(df_final[selected_features])

        # STEP 6: PREDICTION
        prob = model.predict_proba(X_scaled)[0][1]

        result = (
            f"YES – Customer Likely to Churn (Probability: {prob:.2%})"
            if prob >= THRESHOLD
            else f"NO – Customer Likely to Stay (Probability: {prob:.2%})"
        )

        return render_template(
            "index.html",
            prediction_text=result,
            model_name=MODEL_NAME
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Prediction Error: {str(e)}"
        )

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
