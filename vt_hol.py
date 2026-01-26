import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from log_code import setup_logging
logger = setup_logging('vt_hol')

def variable_transformation_outliers(X_train, X_test):
    """
    Variable Transformation + Outlier Handling Tournament
    Goal: Minimize skewness
    """

    X_train_tr = X_train.copy()
    X_test_tr = X_test.copy()
    strategy = {}


    exclude_keywords = ['_hash_', 'target', 'freq', 'count']
    numeric_cols = X_train_tr.select_dtypes(include='number').columns

    continuous_cols = [
        c for c in numeric_cols
        if X_train_tr[c].nunique() > 20
        and not any(k in c.lower() for k in exclude_keywords)
    ]

    logger.info(f"Running VT/Outlier Tournament on {len(continuous_cols)} columns")

    for col in continuous_cols:
        logger.info(f"\nAnalyzing Column: {col}")

        # ---------- OUTLIER CLIPPING ----------
        lower = X_train_tr[col].quantile(0.01)
        upper = X_train_tr[col].quantile(0.99)

        X_train_tr[col] = X_train_tr[col].clip(lower, upper)
        X_test_tr[col] = X_test_tr[col].clip(lower, upper)

        # ---------- TRANSFORMATION TOURNAMENT ----------
        scores = {}

        # 1. Original
        scores['Original'] = abs(X_train_tr[col].skew())

        # 2. Log
        if X_train_tr[col].min() >= 0:
            scores['Log'] = abs(np.log1p(X_train_tr[col]).skew())
        else:
            scores['Log'] = np.inf

        # 3. Sqrt
        if X_train_tr[col].min() >= 0:
            scores['Sqrt'] = abs(np.sqrt(X_train_tr[col]).skew())
        else:
            scores['Sqrt'] = np.inf

        # 4. Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        yj_vals = pt.fit_transform(X_train_tr[[col]])
        scores['YeoJohnson'] = abs(pd.Series(yj_vals.ravel()).skew())

        # 5. Reciprocal
        recip_vals = 1 / (X_train_tr[col] + 1e-4)
        scores['Reciprocal'] = abs(recip_vals.skew())

        # ---------- SELECT WINNER ----------
        best_method = min(scores, key=scores.get)

        logger.info(f"Scores (Absolute Skewness): {scores}")
        logger.info(f"WINNER for '{col}': {best_method}")

        strategy[col] = {
            'method': best_method,
            'scores': scores,
            'outlier_bounds': (lower, upper)
        }

        # ---------- APPLY WINNER ----------
        if best_method == 'Log':
            X_train_tr[col] = np.log1p(X_train_tr[col])
            X_test_tr[col] = np.log1p(X_test_tr[col])

        elif best_method == 'Sqrt':
            X_train_tr[col] = np.sqrt(X_train_tr[col])
            X_test_tr[col] = np.sqrt(X_test_tr[col])

        elif best_method == 'YeoJohnson':
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            X_train_tr[col] = pt.fit_transform(X_train_tr[[col]]).ravel()
            X_test_tr[col] = pt.transform(X_test_tr[[col]]).ravel()
            strategy[col]['model'] = pt

        elif best_method == 'Reciprocal':
            X_train_tr[col] = 1 / (X_train_tr[col] + 1e-4)
            X_test_tr[col] = 1 / (X_test_tr[col] + 1e-4)

        

    return X_train_tr, X_test_tr, strategy

