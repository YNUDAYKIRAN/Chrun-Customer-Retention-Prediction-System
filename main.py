"""
Main execution file for Telecom Churn Prediction Pipeline.
Loads the dataset, applies ML pipeline techniques, and saves the best model.
"""

import numpy as np
import pandas as pd
import sys
import os
import pickle
import warnings
import logging
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import mutual_info_classif

from vt_hol import variable_transformation_outliers
from fs import feature_selection_tournament
from data_scal import scale_and_balance_tournament
from all_models import train_and_select_best


class CHURN_PREDICTION:

    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)

     
            if 'TotalCharges' in self.df.columns:
                self.df['TotalCharges'] = pd.to_numeric(
                    self.df['TotalCharges'], errors='coerce'
                )

            logger.info("Data Loaded Successfully")
            logger.info(f"Shape: {self.df.shape}")
            logger.info(f"Dtypes:\n{self.df.dtypes}")
            logger.info(f"Available Columns: {self.df.columns.tolist()}")

            # Target mapping
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            logger.info(f"Churn Unique Values after mapping: {self.df['Churn'].unique()}")

            # Feature / target split (ONLY ONCE)
            self.X = self.df.drop(columns=['customerID', 'Churn'])
            self.y = self.df['Churn']

            logger.info(f"After X shape: {self.X.shape}")
            logger.info(f"After y shape: {self.y.shape}")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y
            )

            logger.info(f"Original Churn Ratio: {self.y.mean():.2f}")
            logger.info(f"Train Churn Ratio: {self.y_train.mean():.2f}")
            logger.info(f"Test Churn Ratio: {self.y_test.mean():.2f}")

            assert self.X_train['TotalCharges'].dtype != 'object', \"TotalCharges is still object — ingestion fix failed"

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

   
    def missing_values(self):
        try:
            logger.info("=== STEP 2: Missing Value Handling (10-Technique Tournament) ===")

            # 🔴 PERMANENT FIX: enforce numeric BEFORE doing anything
            if 'TotalCharges' in self.X_train.columns:
                self.X_train['TotalCharges'] = pd.to_numeric(
                    self.X_train['TotalCharges'], errors='coerce'
                )
                self.X_test['TotalCharges'] = pd.to_numeric(
                    self.X_test['TotalCharges'], errors='coerce'
                )

            # Use temp copy ONLY for analysis
            X_temp = self.X_train.copy()

            # Identify columns that actually have missing values
            cols_missing = X_temp.columns[X_temp.isnull().any()].tolist()

            if not cols_missing:
                logger.info("No missing values found.")
                return

            for col in cols_missing:

                # ---------- NUMERIC ----------
                if pd.api.types.is_numeric_dtype(X_temp[col]):
                    logger.info(f"Analyzing Column: {col}")

                    numeric_subset = X_temp.select_dtypes(include='number')
                    col_idx = numeric_subset.columns.get_loc(col)
                    var_original = X_temp[col].var()
                    scores = {}

                    fill_mean = X_temp[col].fillna(X_temp[col].mean())
                    scores['Mean'] = abs(var_original - fill_mean.var())

                    fill_median = X_temp[col].fillna(X_temp[col].median())
                    scores['Median'] = abs(var_original - fill_median.var())

                    fill_mode = X_temp[col].fillna(X_temp[col].mode()[0])
                    scores['Mode'] = abs(var_original - fill_mode.var())

                    fill_zero = X_temp[col].fillna(0)
                    scores['Constant_0'] = abs(var_original - fill_zero.var())

                    best_method = min(scores, key=scores.get)
                    logger.info(f"WINNER for '{col}': {best_method}")

                    # APPLY PERMANENTLY
                    if best_method == 'Mean':
                        val = self.X_train[col].mean()
                    elif best_method == 'Median':
                        val = self.X_train[col].median()
                    elif best_method == 'Mode':
                        val = self.X_train[col].mode()[0]
                    else:
                        val = 0

                    self.X_train[col].fillna(val, inplace=True)
                    self.X_test[col].fillna(val, inplace=True)

                # ---------- CATEGORICAL ----------
                else:
                    mode_val = self.X_train[col].mode()[0]
                    self.X_train[col].fillna(mode_val, inplace=True)
                    self.X_test[col].fillna(mode_val, inplace=True)
                    logger.info(f"Categorical '{col}': Defaulted to Mode")

            logger.info("Missing value handling complete.")
            logger.info(f"TotalCharges dtype after missing handling: {self.X_train['TotalCharges'].dtype}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")


   
    def cat_num(self):
        try:
            logger.info("=== STEP 3: Grand Slam Categorical Encoding (FINAL FIXED v2) ===")

            X_train = self.X_train.copy()
            X_test = self.X_test.copy()
            y_train = self.y_train.copy()

            self.encoding_strategy = {}
            self.encoders = {}

            numeric_force_cols = ['TotalCharges', 'MonthlyCharges', 'tenure']
            for col in numeric_force_cols:
                if col in X_train.columns:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

         
            logger.info(f"TotalCharges dtype before encoding: {X_train['TotalCharges'].dtype}")
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            new_train_parts = []
            new_test_parts = []

            for col in cat_cols:
                logger.info(f"\nEvaluating techniques for: {col}")
                scores = {}

                # -------- TOURNAMENT --------
                le = LabelEncoder()
                le_enc = le.fit_transform(
                    X_train[col].fillna("Missing").astype(str)
                ).reshape(-1, 1)
                scores['Label'] = mutual_info_classif(le_enc, y_train, random_state=42)[0]

                freq_map = X_train[col].value_counts(normalize=True).to_dict()
                freq_enc = X_train[col].map(freq_map).fillna(0).values.reshape(-1, 1)
                scores['Frequency'] = mutual_info_classif(freq_enc, y_train, random_state=42)[0]

                count_map = X_train[col].value_counts().to_dict()
                count_enc = X_train[col].map(count_map).fillna(0).values.reshape(-1, 1)
                scores['Count'] = mutual_info_classif(count_enc, y_train, random_state=42)[0]

                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                ohe_enc = ohe.fit_transform(X_train[[col]])
                scores['OneHot'] = np.mean(
                    mutual_info_classif(ohe_enc, y_train, random_state=42)
                )

                tm_map = y_train.groupby(X_train[col]).mean().to_dict()
                tm_enc = X_train[col].map(tm_map).fillna(y_train.mean()).values.reshape(-1, 1)
                scores['TargetMean'] = mutual_info_classif(tm_enc, y_train, random_state=42)[0]

                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                oe_enc = oe.fit_transform(X_train[[col]])
                scores['Ordinal'] = mutual_info_classif(oe_enc, y_train, random_state=42)[0]

                hasher = FeatureHasher(n_features=5, input_type='string')
                hash_enc = hasher.transform(
                    X_train[col].astype(str).apply(lambda x: [x])
                ).toarray()
                scores['Hashing'] = np.mean(
                    mutual_info_classif(hash_enc, y_train, random_state=42)
                )

                # -------- SELECT WINNER --------
                best_method = max(scores, key=scores.get)
                self.encoding_strategy[col] = best_method
                logger.info(f"WINNER -> {col}: {best_method}")

                # -------- APPLY WINNER --------
                if best_method == 'Label':
                    le = LabelEncoder()
                    tr = le.fit_transform(X_train[col].fillna("Missing").astype(str))
                    te = X_test[col].fillna("Missing").astype(str)
                    te = te.map(lambda x: x if x in le.classes_ else "Missing")
                    te = le.transform(te)

                    new_train_parts.append(pd.DataFrame({col: tr}, index=X_train.index))
                    new_test_parts.append(pd.DataFrame({col: te}, index=X_test.index))
                    self.encoders[col] = le

                elif best_method in ['Frequency', 'Count']:
                    normalize = best_method == 'Frequency'
                    mapper = X_train[col].value_counts(normalize=normalize).to_dict()

                    new_train_parts.append(
                        pd.DataFrame({col: X_train[col].map(mapper).fillna(0)}, index=X_train.index)
                    )
                    new_test_parts.append(
                        pd.DataFrame({col: X_test[col].map(mapper).fillna(0)}, index=X_test.index)
                    )
                    self.encoders[col] = mapper

                elif best_method == 'TargetMean':
                    mapper = y_train.groupby(X_train[col]).mean().to_dict()
                    global_val = y_train.mean()

                    new_train_parts.append(
                        pd.DataFrame({col: X_train[col].map(mapper).fillna(global_val)}, index=X_train.index)
                    )
                    new_test_parts.append(
                        pd.DataFrame({col: X_test[col].map(mapper).fillna(global_val)}, index=X_test.index)
                    )
                    self.encoders[col] = {'map': mapper, 'fill': global_val}

                elif best_method == 'OneHot':
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    tr = ohe.fit_transform(X_train[[col]])
                    te = ohe.transform(X_test[[col]])

                    cols = ohe.get_feature_names_out([col])
                    new_train_parts.append(pd.DataFrame(tr, columns=cols, index=X_train.index))
                    new_test_parts.append(pd.DataFrame(te, columns=cols, index=X_test.index))
                    self.encoders[col] = ohe

                elif best_method == 'Ordinal':
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    tr = oe.fit_transform(X_train[[col]]).ravel()
                    te = oe.transform(X_test[[col]]).ravel()

                    new_train_parts.append(pd.DataFrame({col: tr}, index=X_train.index))
                    new_test_parts.append(pd.DataFrame({col: te}, index=X_test.index))
                    self.encoders[col] = oe

                elif best_method == 'Hashing':
                    hasher = FeatureHasher(n_features=5, input_type='string')
                    tr = hasher.transform(
                        X_train[col].astype(str).apply(lambda x: [x])
                    ).toarray()
                    te = hasher.transform(
                        X_test[col].astype(str).apply(lambda x: [x])
                    ).toarray()

                    cols = [f"{col}_hash_{i}" for i in range(5)]
                    new_train_parts.append(pd.DataFrame(tr, columns=cols, index=X_train.index))
                    new_test_parts.append(pd.DataFrame(te, columns=cols, index=X_test.index))
                    self.encoders[col] = hasher

            # -------- FINAL ASSEMBLY --------
            X_train_num = X_train.drop(columns=cat_cols)
            X_test_num = X_test.drop(columns=cat_cols)

            self.X_train = pd.concat([X_train_num] + new_train_parts, axis=1)
            self.X_test = pd.concat([X_test_num] + new_test_parts, axis=1)

            pickle.dump(self.encoders, open("encoders.pkl", "wb"))

            logger.info(f"Shape: {self.df.shape}")
            logger.info(f"Dtypes:{self.df.dtypes}")
            logger.info(f"Available Columns: {self.df.columns.tolist()}")
            logger.info(f"After X shape: {self.X.shape}")



        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")



    def vt_hol(self):
        try:
            logger.info("=== STEP 4: Variable Transformation & Outliers (Tournament) ===")

            X_train_tr, X_test_tr, vt_strategy = variable_transformation_outliers(
                self.X_train, self.X_test
            )

            self.X_train = X_train_tr
            self.X_test = X_test_tr
            self.vt_strategy = vt_strategy  # 🔴 STORE IT

            assert list(self.X_train.columns) == list(self.X_test.columns), \
                "Column mismatch after VT/Outlier handling"

            logger.info(f"After VT/OH Shape: {self.X_train.shape}")
            logger.info(f"X_train shape after encoding: {self.X_train.shape}")
            logger.info(f"X_test shape after encoding: {self.X_test.shape}")

            logger.info("Variable Transformation applied successfully.")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")


    def feature_selection(self):
        try:
            logger.info("=== STEP 5: Feature Selection (8-Strategy Tournament) ===")

            X_train_fs, X_test_fs, selected_features = feature_selection_tournament(
                self.X_train,
                self.X_test,
                self.y_train
            )

            self.selected_features = list(selected_features)
            self.X_train = X_train_fs[self.selected_features].copy()
            self.X_test = X_test_fs[self.selected_features].copy()

            with open("selected_features.pkl", "wb") as f:
                pickle.dump(self.selected_features, f)

            logger.info(f"Final Selected Feature Count: {len(self.selected_features)}")
            logger.info(f"X_train FS Shape: {self.X_train.shape}")
            logger.info(f"X_test FS Shape: {self.X_test.shape}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(f"Error in Line no : {er_line.tb_lineno} :due to {er_msg}")

    def scaling_and_balancing(self):
        try:
            logger.info("=== STEP 6 & 7: Scaling + Balancing (Grand Slam Tournament) ===")

            self.X_train = self.X_train[self.selected_features].copy()
            self.X_test = self.X_test[self.selected_features].copy()

            logger.info(f"Locked features count: {len(self.selected_features)}")
            logger.info(f"X_train shape before scaling: {self.X_train.shape}")
            logger.info(f"X_test shape before scaling: {self.X_test.shape}")
            self.X_train_bal, self.y_train_bal, \
                self.X_test_bal, self.y_test_bal, \
                self.scaler = scale_and_balance_tournament(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.selected_features
            )

            with open("scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            logger.info("Grand Slam Tournament Complete. Best methods applied.")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(
                f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}"
            )



  


    def all_models(self):
        try:
            logger.info("=== FINAL STEP: MODEL TRAINING GRAND PRIX ===")

            # Ensure y_test is integer
            self.y_test = self.y_test.astype(int)

            best_model_name, best_auc, results_df = train_and_select_best(
                self.X_train_bal,  # balanced + scaled train
                self.y_train_bal,
                self.X_test_bal,  # scaled test (NO leakage)
                self.y_test,
                model_path="best_model.pkl"
            )

            logger.info("\n=== FINAL LEADERBOARD ===")
            logger.info(f"\n{results_df.to_string(index=False)}")

            # Save leaderboard
            results_df.to_csv("model_leaderboard.csv", index=False)
            logger.info("Leaderboard saved to 'model_leaderboard.csv'")

            # Save deployment artifacts
            import pickle
            with open("scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            with open("selected_features.pkl", "wb") as f:
                pickle.dump(self.selected_features, f)

            with open("encoding_strategy.pkl", "wb") as f:
                pickle.dump(self.encoding_strategy, f)

            logger.info("All Deployment Artifacts Saved Successfully.")

            # Final winner log
            logger.info(
                f" FINAL WINNER MODEL: {best_model_name} | AUC = {best_auc:.4f}"
            )

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.error(
                f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}"
            )


if __name__ == "__main__":
    try:
        obj = CHURN_PREDICTION(f"D:\\DATA_SCIENCE WITH AI\\Internship\\Task_1_Teleco\\archive (2)\\Updated_Churn.csv")

        obj.missing_values()
        obj.cat_num()
        obj.vt_hol()
        obj.feature_selection()
        obj.scaling_and_balancing()
        obj.all_models()


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.error(f"Error in Line no : {er_line.tb_lineno} :due to {er_msg}")

