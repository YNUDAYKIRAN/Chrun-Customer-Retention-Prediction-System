import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ===============================
# LOAD ARTIFACTS
# ===============================
PKL_PATH = r"D:\\DATA_SCIENCE WITH AI\\Internship\\Task_1_Teleco\\pkl"

def load_pkl(filename):
    with open(os.path.join(PKL_PATH, filename), "rb") as f:
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



@app.route("/predict", methods=["POST"])
def predict():
    try:
       
        input_data = dict(request.form)
        df = pd.DataFrame([input_data])

        print("FORM DATA RECEIVED:")
        print(df)

        numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

     
        encoded_values = {}

        for col, method in encoding_strategy.items():

            if col not in df.columns:
                continue

            value = df[col].iloc[0]

            # -------- ONE HOT --------
            if method == "OneHot":
                ohe = encoders[col]
                arr = ohe.transform(df[[col]])
                ohe_cols = ohe.get_feature_names_out([col])

                for i, c in enumerate(ohe_cols):
                    encoded_values[c] = arr[0][i]

            # -------- LABEL ENCODING --------
            elif method == "Label":
                le = encoders[col]
                if value not in le.classes_:
                    value = le.classes_[0]
                encoded_values[col] = le.transform([value])[0]

            # -------- FREQUENCY / COUNT --------
            elif method in ["Frequency", "Count"]:
                mapper = encoders[col]
                encoded_values[col] = mapper.get(value, 0)

            # -------- TARGET MEAN --------
            elif method == "TargetMean":
                mapper = encoders[col]["map"]
                fill = encoders[col]["fill"]
                encoded_values[col] = mapper.get(value, fill)

            # -------- ORDINAL --------
            elif method == "Ordinal":
                oe = encoders[col]
                encoded_values[col] = oe.transform([[value]])[0][0]

            # -------- HASHING --------
            elif method == "Hashing":
                hasher = encoders[col]
                arr = hasher.transform([[str(value)]]).toarray()
                for i in range(arr.shape[1]):
                    encoded_values[f"{col}_hash_{i}"] = arr[0][i]

  
        df_final = pd.DataFrame(0.0, index=[0], columns=selected_features)

        for col in selected_features:
            if col in encoded_values:
                df_final[col] = encoded_values[col]
            elif col in df.columns:
                df_final[col] = df[col].iloc[0]

   
        X_scaled = scaler.transform(df_final[selected_features])

      
        prob = model.predict_proba(X_scaled)[0][1]

        result = (
            f"YES – Customer Likely to Churn (Probability: {prob:.2%})"
            if prob >= THRESHOLD
            else f"NO – Customer Likely to Stay (Probability: {prob:.2%})"
        )

        print("PREDICTION:", result)

        # 🔴 IMPORTANT: RETURN SAME TEMPLATE WITH RESULT
        return render_template(
            "index.html",
            prediction_text=result,
            model_name=MODEL_NAME
        )

    except Exception as e:
        print("ERROR:", str(e))
        return render_template(
            "index.html",
            prediction_text=f"Prediction Error: {str(e)}"
        )



if __name__ == "__main__":
    app.run(debug=True)
