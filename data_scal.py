import pandas as pd
import numpy as np
import pickle
import sys

# --- SCALING IMPORTS ---
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer
)

# --- BALANCING IMPORTS ---
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
)
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, ClusterCentroids
)
from imblearn.combine import SMOTEENN, SMOTETomek

# --- MODEL IMPORTS ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from log_code import setup_logging

logger = setup_logging("data_scal")


def scale_and_balance_tournament(
    X_train, y_train, X_test, y_test, selected_features
):
    """
    STEP 6 & 7: SCALING + BALANCING GRAND SLAM TOURNAMENT
    (FIXED: Scaling happens ONLY on selected features)
    """
    try:
        logger.info("=== STEP 6 & 7: Scaling + Balancing Tournament ===")
        X_train = X_train[selected_features].copy()
        X_test = X_test[selected_features].copy()

        logger.info(
            f"Locked feature count before scaling: {len(selected_features)}"
        )

        logger.info("\n--- ROUND 1: SCALING TOURNAMENT ---")

        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler(),
            "Normalizer_L2": Normalizer(norm="l2"),
            "Power_YeoJohnson": PowerTransformer(method="yeo-johnson"),
            "Quantile_Normal": QuantileTransformer(
                output_distribution="normal", random_state=42
            ),
            "Quantile_Uniform": QuantileTransformer(
                output_distribution="uniform", random_state=42
            ),
            "Power_BoxCox": PowerTransformer(method="box-cox"),
        }

        judge = LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=42
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        scaler_scores = {}

        for name, scaler in scalers.items():
            try:
                X_temp = X_train.copy()

                # Handle BoxCox positivity
                if "BoxCox" in name and (X_temp <= 0).any().any():
                    shift = abs(X_temp.min().min()) + 1
                    X_temp = X_temp + shift

                X_temp_scaled = pd.DataFrame(
                    scaler.fit_transform(X_temp),
                    columns=X_temp.columns,
                    index=X_temp.index
                )

                scores = cross_val_score(
                    judge, X_temp_scaled, y_train,
                    cv=cv, scoring="roc_auc"
                )
                scaler_scores[name] = scores.mean()

            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                scaler_scores[name] = -1

        logger.info("\n=== SCALING TOURNAMENT RESULTS ===")
        for k, v in sorted(
            scaler_scores.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"{k.ljust(25)} | ROC_AUC = {v:.4f}")

        best_scaler_name = max(scaler_scores, key=scaler_scores.get)
        best_scaler = scalers[best_scaler_name]

        logger.info(
            f"\nSCALING WINNER -> {best_scaler_name} "
            f"(ROC_AUC = {scaler_scores[best_scaler_name]:.4f})"
        )

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        if "BoxCox" in best_scaler_name and (X_train_scaled <= 0).any().any():
            shift = abs(X_train_scaled.min().min()) + 1
            X_train_scaled += shift
            X_test_scaled += shift

        best_scaler.fit(X_train_scaled)

        X_train_scaled = pd.DataFrame(
            best_scaler.transform(X_train_scaled),
            columns=X_train_scaled.columns,
            index=X_train_scaled.index
        )

        X_test_scaled = pd.DataFrame(
            best_scaler.transform(X_test_scaled),
            columns=X_test_scaled.columns,
            index=X_test_scaled.index
        )

     
        logger.info("\n--- ROUND 2: BALANCING TOURNAMENT ---")

        balancers = {
            "ClassWeight": None,
            "RandomOverSampler": RandomOverSampler(random_state=42),
            "SMOTE": SMOTE(random_state=42),
            "ADASYN": ADASYN(random_state=42),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
            "SVMSMOTE": SVMSMOTE(random_state=42),
            "RandomUnderSampler": RandomUnderSampler(random_state=42),
            "NearMiss": NearMiss(version=1),
            "SMOTEENN": SMOTEENN(random_state=42),
            "SMOTETomek": SMOTETomek(random_state=42),
            "ClusterCentroids": ClusterCentroids(random_state=42),
        }

        balance_scores = {}

        for name, sampler in balancers.items():
            try:
                if sampler is None:
                    X_res, y_res = X_train_scaled, y_train
                    model = LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=42,
                    )
                else:
                    X_res, y_res = sampler.fit_resample(
                        X_train_scaled, y_train
                    )
                    model = LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=42,
                    )

                scores = cross_val_score(
                    model, X_res, y_res,
                    cv=cv, scoring="roc_auc"
                )
                balance_scores[name] = scores.mean()

            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                balance_scores[name] = -1

        logger.info("\n=== BALANCING TOURNAMENT RESULTS ===")
        for k, v in sorted(
            balance_scores.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"{k.ljust(25)} | ROC_AUC = {v:.4f}")

        best_balance_name = max(balance_scores, key=balance_scores.get)
        logger.info(
            f"\nBALANCING WINNER -> {best_balance_name} "
            f"(ROC_AUC = {balance_scores[best_balance_name]:.4f})"
        )

        best_sampler = balancers[best_balance_name]

        if best_sampler is None:
            X_train_final, y_train_final = X_train_scaled, y_train
        else:
            X_train_final, y_train_final = best_sampler.fit_resample(
                X_train_scaled, y_train
            )

        logger.info(f"Final Train Shape: {X_train_final.shape}")
        logger.info(
            f"Final Class Distribution: "
            f"{y_train_final.value_counts().to_dict()}"
        )

        return (
            X_train_final,
            y_train_final,
            X_test_scaled,
            y_test,
            best_scaler,
        )

    except Exception as e:
        t, m, line = sys.exc_info()
        logger.error(
            f"Scaling/Balancing Error at line {line.tb_lineno}: {m}"
        )
        raise

