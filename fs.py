import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from log_code import setup_logging

logger = setup_logging('fs')


def feature_selection_tournament(X_train, X_test, y_train):
    """
    GRAND SLAM FEATURE SELECTION TOURNAMENT
    """
    try:


        initial_count = X_train.shape[1]

        vt = VarianceThreshold(0.01)
        vt.fit(X_train)

        X_train = X_train.loc[:, vt.get_support()]
        X_test = X_test.loc[:, vt.get_support()]

        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]

        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)

        current_cols = np.array(X_train.columns)

        logger.info(
            f"Cleaned Features: {len(current_cols)} "
            f"(Dropped {initial_count - len(current_cols)})"
        )

       
        k_target = min(20, int(len(current_cols) * 0.75))
        logger.info(f"Targeting Top {k_target} features per strategy")

        strategies = {}

        # ---------- ANOVA ----------
        sel = SelectKBest(f_classif, k=k_target)
        sel.fit(X_train, y_train)
        strategies['ANOVA'] = current_cols[sel.get_support()]

        # ---------- Mutual Info ----------
        sel = SelectKBest(mutual_info_classif, k=k_target)
        sel.fit(X_train, y_train)
        strategies['MutualInfo'] = current_cols[sel.get_support()]

        # ---------- RFE Logistic (FIXED) ----------
        sel = RFE(
            estimator=LogisticRegression(max_iter=1000),
            n_features_to_select=k_target,
            step=0.2
        )
        sel.fit(X_train, y_train)
        strategies['RFE_Logistic'] = current_cols[sel.get_support()]

        # ---------- RFE Tree (FIXED) ----------
        sel = RFE(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_features_to_select=k_target,
            step=0.2
        )
        sel.fit(X_train, y_train)
        strategies['RFE_Tree'] = current_cols[sel.get_support()]

        # ---------- Lasso ----------
        sel = SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
        )
        sel.fit(X_train, y_train)
        cols = current_cols[sel.get_support()]
        strategies['Lasso_L1'] = cols if len(cols) else current_cols[:k_target]

        # ---------- Random Forest ----------
        sel = SelectFromModel(
            RandomForestClassifier(n_estimators=50, random_state=42)
        )
        sel.fit(X_train, y_train)
        cols = current_cols[sel.get_support()]
        strategies['RandomForest'] = cols if len(cols) else current_cols[:k_target]

        # ---------- Extra Trees ----------
        sel = SelectFromModel(
            ExtraTreesClassifier(n_estimators=50, random_state=42)
        )
        sel.fit(X_train, y_train)
        cols = current_cols[sel.get_support()]
        strategies['ExtraTrees'] = cols if len(cols) else current_cols[:k_target]

        # ---------- Gradient Boost ----------
        sel = SelectFromModel(
            GradientBoostingClassifier(n_estimators=50, random_state=42)
        )
        sel.fit(X_train, y_train)
        cols = current_cols[sel.get_support()]
        strategies['GradBoost'] = cols if len(cols) else current_cols[:k_target]

      
        logger.info("\n=== FEATURE SELECTION RESULTS (CV AUC) ===")

        results = {}
        validator = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for name, cols in strategies.items():
            X_subset = X_train[cols]
            scores = cross_val_score(
                validator, X_subset, y_train,
                cv=cv, scoring='roc_auc'
            )
            mean_auc = scores.mean()
            results[name] = mean_auc

            logger.info(
                f"{name.ljust(15)} | AUC: {mean_auc:.4f} | Features: {len(cols)}"
            )

      
        best_strategy = max(results, key=results.get)
        best_features = list(strategies[best_strategy])

        logger.info("\n=== FEATURE SELECTION WINNER ===")
        logger.info(f"WINNER STRATEGY : {best_strategy}")
        logger.info(f"BEST AUC SCORE  : {results[best_strategy]:.4f}")
        logger.info(f"FEATURE COUNT  : {len(best_features)}")

      
        ALWAYS_KEEP = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'PaymentMethod'
        ]

        rescued = 0
        for key in ALWAYS_KEEP:
            for col in current_cols:
                if key in col and col not in best_features:
                    best_features.append(col)
                    rescued += 1

        if rescued:
            logger.info(f"+ Rescued {rescued} Business-Critical Features")

        best_features = list(set(best_features))

        X_train_final = X_train[best_features]
        X_test_final = X_test[best_features]

        logger.info(f"FINAL FEATURE COUNT: {len(best_features)}")

        with open("selected_features.pkl", "wb") as f:
            pickle.dump(best_features, f)

        return X_train_final, X_test_final, best_features

    except Exception as e:
        t, m, line = sys.exc_info()
        logger.error(f"FS Tournament Error at Line {line.tb_lineno}: {str(e)}")


