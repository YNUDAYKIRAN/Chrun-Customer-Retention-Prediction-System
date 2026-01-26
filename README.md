📊 AI-Powered Telecom Customer Churn Prediction System

An end-to-end Machine Learning–driven Customer Churn Prediction System designed to identify high-risk customers and enable proactive retention strategies.
This project implements a fully automated ML pipeline with tournament-based decision making at every stage — from preprocessing to model selection — and includes a Flask web application for real-time predictions.

🚀 Project Highlights

🔁 Tournament-based ML pipeline (best technique wins at every step)
🧠 Multiple ML models trained & evaluated automatically
📈 ROC-AUC driven model selection
🧪 Advanced feature engineering & selection
⚖️ Automated class imbalance handling
🌐 Production-ready Flask deployment
🔒 Strict feature locking to prevent data leakage
📊 Rich EDA with 30+ professional visualizations
🏢 Business Problem

Customer churn directly impacts revenue and growth. Acquiring a new customer costs 5–7x more than retaining an existing one.

This system helps businesses:
Detect early churn signals

Segment high-risk customers

Take data-driven retention actions

Reduce revenue leakage proactively

📂 Dataset

Source: Telco Customer Churn Dataset (Kaggle)

Records: 7,043 customers

Target Variable: Churn (Yes / No)

Key Feature Groups

Demographics (gender, senior citizen, dependents)

Services (internet, phone, add-ons)

Billing & contracts

Usage & tenure

Synthetic segmentation features (region, device, network, usage pattern)

🧠 System Architecture
Data Ingestion
      ↓
Missing Value Tournament
      ↓
Categorical Encoding Tournament
      ↓
Outlier Handling + Variable Transformation
      ↓
Feature Selection Tournament
      ↓
Scaling Tournament
      ↓
Balancing Tournament
      ↓
Model Grand Prix (9 Models)
      ↓
Best Model Selection (ROC-AUC)
      ↓
Flask Deployment

🧪 Machine Learning Pipeline (Step-by-Step)
✅ 1. Missing Value Handling

Mean / Median / Mode / Constant comparison

Variance preservation used to select the best method

✅ 2. Categorical Encoding Tournament

Techniques evaluated per feature:

Label Encoding

Frequency Encoding

Count Encoding

Target Mean Encoding

One-Hot Encoding

Ordinal Encoding

Feature Hashing

➡ Winner chosen using Mutual Information Score

✅ 3. Outlier Handling & Variable Transformation

1%–99% clipping

Transformation tournament:

Log

Square Root

Yeo-Johnson

Reciprocal

Original

➡ Lowest skewness wins

✅ 4. Feature Selection (Grand Slam)

Methods used:

ANOVA

Mutual Information

RFE (Logistic & Tree)

L1 (Lasso)

Random Forest

Extra Trees

Gradient Boosting

➡ Best subset chosen via cross-validated ROC-AUC
➡ Business-critical features are force-retained

✅ 5. Scaling & Balancing Tournament

Scalers Tested

StandardScaler

MinMaxScaler

RobustScaler

PowerTransformer

QuantileTransformer

Normalizer

Balancing Methods

Class Weights

SMOTE / ADASYN / BorderlineSMOTE

Under-sampling

Hybrid methods (SMOTEENN, SMOTETomek)

➡ Best combination selected via ROC-AUC

✅ 6. Model Training (Grand Prix)

Models evaluated:

Logistic Regression

KNN

Naive Bayes

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

SVM

📊 Metrics:

Accuracy

ROC-AUC

Confusion Matrix

Classification Report

Optimal Threshold (Youden’s J)

➡ Best model auto-saved

📊 Exploratory Data Analysis (EDA)

Churn distribution

Tenure vs churn

Monthly charges analysis

Contract & billing behavior

Value-added services impact

Senior citizen behavior

Multi-dimensional segmentation

Correlation heatmap

📁 All plots saved automatically for reporting.

🌐 Web Application (Flask)
Features

User-friendly form interface

Strict feature alignment with training pipeline

No unseen categories allowed

Probability-based churn prediction

Threshold-aware decision logic

Prediction Output
YES – Customer Likely to Churn (Prob: 0.78)
NO – Customer Likely to Stay (Prob: 0.21)

📦 Project Structure
├── app.py                     # Flask application
├── main.py                    # Complete ML pipeline
├── vt_hol.py                  # Outlier + transformation tournament
├── fs.py                      # Feature selection tournament
├── data_scal.py               # Scaling & balancing tournament
├── all_models.py              # Model training & evaluation
├── Visual_Prt_Data.py         # EDA & visualization module
├── log_code.py                # Centralized logging
├── encoders.pkl               # Saved encoders
├── scaler.pkl                 # Best scaler
├── selected_features.pkl      # Locked feature list
├── best_model.pkl             # Final trained model
├── model_leaderboard.csv      # Model comparison
├── roc_curve_comparison.png
├── logs/
└── templates/
    └── index.html

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Model
python main.py

3️⃣ Run Web App
python app.py


Open browser:

http://127.0.0.1:5000/

📈 Business Impact

Early churn detection

Targeted retention campaigns

Reduced acquisition cost

Improved customer lifetime value

Production-ready decision system

🔮 Future Enhancements

SHAP / LIME explainability

CRM integration

Automated retention offers

Deep learning (LSTM churn modeling)

Real-time dashboards (Streamlit)

👨‍💻 Author

Uday Kiran
Data Science & Machine Learning Engineer
🎯 Focus: End-to-End ML Systems | Production Pipelines | Applied AI
