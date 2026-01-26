from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


# Best Model + Threshold
with open("best_model.pkl", "rb") as f:
    artifact = pickle.load(f)
    model = artifact["model"]
    THRESHOLD = artifact["threshold"]
    MODEL_NAME = artifact.get("name", "BestModel")

# Scaler (trained AFTER feature selection)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Final selected features (18)
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Encoding strategy
with open("encoding_strategy.pkl", "rb") as f:
    encoding_strategy = pickle.load(f)

# Encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:

        input_data = dict(request.form)
        df = pd.DataFrame([input_data])
        
        rename_map = {
            "DeviceType": "Device_Type",
            "NetworkType": "Network_Type",
            "UsagePattern": "Usage_Pattern"
        }
        df.rename(columns=rename_map, inplace=True)

        numeric_cols = ["SeniorCitizen", "tenure",
                        "MonthlyCharges", "TotalCharges"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col, method in encoding_strategy.items():

            # Only handle OneHot columns
            if method != "OneHot":
                continue

            # Column must exist in input
            if col not in df.columns:
                continue

            # Encoder must exist
            if col not in encoders:
                continue

            ohe = encoders[col]

            # Transform input (safe even if unseen category)
            arr = ohe.transform(df[[col]])

            ohe_cols = ohe.get_feature_names_out([col])

            df_ohe = pd.DataFrame(
                arr,
                columns=ohe_cols,
                index=df.index
            )

            # Activate ONLY selected OneHot features
            for c in ohe_cols:
                if c in selected_features:
                    df[c] = pd.to_numeric(df_ohe[c], errors="coerce").fillna(0)


        df_final = pd.DataFrame(
            0.0,  # force float
            index=[0],
            columns=selected_features
        )

        for col in df.columns:
            if col in df_final.columns:
                # convert safely to numeric
                df_final[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).values



  
        X_scaled = scaler.transform(df_final)
        
        prob = model.predict_proba(X_scaled)[0][1]

        if prob >= THRESHOLD:
            result = f"YES – Customer Likely to Churn "
        else:
            result = f"NO – Customer Likely to Stay "

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

if __name__ == "__main__":
    app.run(debug=True)

