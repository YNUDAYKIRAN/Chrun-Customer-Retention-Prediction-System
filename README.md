# 📊 AI-Powered Telecom Customer Churn Prediction System

An end-to-end **Machine Learning–driven Customer Churn Prediction System** designed to identify customers at risk of leaving and enable proactive retention strategies.

This project implements a **fully automated, tournament-based ML pipeline**, where the best-performing technique is selected at every stage — from preprocessing to model selection — and is deployed using a **Flask web application**.

---

## 🚀 Project Highlights

- Tournament-based decision making at every ML stage  
- Automated feature engineering and selection  
- ROC-AUC driven model comparison  
- Robust handling of class imbalance  
- End-to-end production-ready ML pipeline  
- Flask-based web application for real-time predictions  
- Strict feature locking to avoid data leakage  
- Professional EDA with saved visual reports  

---

## 🏢 Business Problem

Customer churn is a major challenge in competitive industries like telecom, banking, and SaaS.

- Acquiring a new customer costs 5–7x more than retaining an existing one  
- Businesses often lack early warning signals for churn  
- Reactive retention strategies lead to revenue loss  

This system helps organizations **predict churn in advance** and take **data-driven retention actions**.

---

## 📂 Dataset

- **Dataset**: Telco Customer Churn (Kaggle)  
- **Records**: 7,043 customers  
- **Target Variable**: `Churn` (Yes / No)

### Feature Categories
- Customer demographics  
- Subscription and service details  
- Billing and contract information  
- Usage behavior and tenure  
- Synthetic segmentation features for advanced analysis  

---

## 🧠 System Architecture

Data Ingestion  
→ Missing Value Handling Tournament  
→ Categorical Encoding Tournament  
→ Outlier Handling & Variable Transformation  
→ Feature Selection Tournament  
→ Scaling Tournament  
→ Balancing Tournament  
→ Model Training Grand Prix  
→ Best Model Selection  
→ Flask Deployment  

---

## 🧪 Machine Learning Pipeline

### 1️⃣ Missing Value Handling
- Mean, Median, Mode, Constant comparison  
- Best strategy selected using variance preservation  

### 2️⃣ Categorical Encoding Tournament
Techniques evaluated per feature:
- Label Encoding  
- Frequency Encoding  
- Count Encoding  
- Target Mean Encoding  
- One-Hot Encoding  
- Ordinal Encoding  
- Feature Hashing  

Winner selected using **Mutual Information Score**.

---

### 3️⃣ Outlier Handling & Variable Transformation
- Outlier clipping (1st–99th percentile)  
- Transformations tested:
  - Log  
  - Square Root  
  - Yeo-Johnson  
  - Reciprocal  
  - Original  

Winner chosen based on **minimum skewness**.

---

### 4️⃣ Feature Selection (Grand Slam)
Methods used:
- ANOVA  
- Mutual Information  
- Recursive Feature Elimination  
- L1 (Lasso)  
- Random Forest  
- Extra Trees  
- Gradient Boosting  

Best feature subset selected using **cross-validated ROC-AUC**, with business-critical features preserved.

---

### 5️⃣ Scaling & Balancing Tournament

**Scalers Evaluated**
- StandardScaler  
- MinMaxScaler  
- RobustScaler  
- PowerTransformer  
- QuantileTransformer  

**Balancing Techniques**
- Class Weights  
- SMOTE  
- ADASYN  
- Borderline-SMOTE  
- Under-sampling  
- Hybrid methods  

Best scaler + balancing strategy selected via ROC-AUC.

---

### 6️⃣ Model Training (Grand Prix)

Models trained and evaluated:
- Logistic Regression  
- KNN  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- Support Vector Machine  

Metrics used:
- Accuracy  
- ROC-AUC  
- Confusion Matrix  
- Classification Report  
- Optimal Decision Threshold  

Best-performing model is automatically saved.

---

## 📊 Exploratory Data Analysis (EDA)

- Churn distribution analysis  
- Tenure vs churn  
- Monthly charges impact  
- Contract and billing behavior  
- Value-added services effect  
- Senior citizen churn trends  
- Multi-dimensional segmentation  
- Correlation heatmap  

All plots are generated and saved automatically.

---

## 🌐 Web Application (Flask)

### Features
- User-friendly input form  
- Strict feature alignment with training pipeline  
- No unseen categories allowed  
- Probability-based churn prediction  
- Threshold-aware decision logic  

### Prediction Output
YES – Customer Likely to Churn (Prob: 0.78)  
NO – Customer Likely to Stay (Prob: 0.21)  

---

## 📦 Project Structure

## 📁 Project Structure

<pre>
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   └── processed/
│
├── src/
│   ├── main.py
│   ├── vt_hol.py
│   ├── fs.py
│   ├── data_scal.py
│   ├── all_models.py
│   ├── Visual_Prt_Data.py
│   └── log_code.py
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── selected_features.pkl
│
├── reports/
│   ├── model_leaderboard.csv
│   └── roc_curve_comparison.png
│
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
</pre>




---

## ▶️ How to Run the Project

### Step 1: Install Dependencies


pip install -r requirements.txt


### Step 2: Train the Model


python main.py


### Step 3: Run the Web Application


python app.py


Open in browser:


http://127.0.0.1:5000/


🚀 Live Demo (Deployed Application)

🔗 Live Application URL:
👉 https://ai-powered-customer-retention-prediction-nvyo.onrender.com

The application is deployed on Render and allows users to enter customer details and instantly receive:

Churn Risk (Yes / No)

## 📈 Business Impact

- Early identification of at-risk customers  
- Targeted retention strategies  
- Reduced customer acquisition cost  
- Improved customer lifetime value  
- Production-ready decision support system  

---

## 🔮 Future Enhancements

- SHAP / LIME explainability  
- CRM integration  
- Automated retention campaigns  
- Deep learning models (LSTM)  
- Interactive dashboards  

---

## 👨‍💻 Author

**Uday Kiran**  
Data Science & Machine Learning Engineer  
Focus: End-to-End ML Systems | Applied AI | Production Pipelines  
