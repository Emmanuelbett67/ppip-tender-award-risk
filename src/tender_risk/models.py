"""Model families, hyperparameter spaces and threshold selection."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED

BASE_MODELS = {
    "Logistic regression": (LogisticRegression(max_iter=3000),
                            {"C": np.logspace(-3, 1, 20)}),
    "Random forest": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
                      {"max_depth": [12, 18, 24, None], "min_samples_leaf": [1, 3, 5, 10],
                       "max_features": ["sqrt", 0.3]}),
    "Gradient boosting": (HistGradientBoostingClassifier(max_iter=400, random_state=SEED),
                          {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [31, 63, 127],
                           "l2_regularization": [0.0, 0.1, 1.0], "min_samples_leaf": [20, 50, 100]}),
    "XGBoost": (xgb.XGBClassifier(n_estimators=400, tree_method="hist", n_jobs=-1, random_state=SEED,
                                  eval_metric="logloss"),
                {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
                 "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10], "reg_lambda": [1, 5]}),
}
MODEL_NAMES = list(BASE_MODELS) + ["Stacked ensemble"]

# Hyperparameters selected by randomised search on 2018–2019 -> 2020 (scripts/evaluate.py --tune reproduces them).
DEFAULT_PARAMS = {
    "Logistic regression": {"C": 0.04832930238571752},
    "Random forest": {"min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 12},
    "Gradient boosting": {"min_samples_leaf": 50, "max_leaf_nodes": 63, "learning_rate": 0.03, "l2_regularization": 0.0},
    "XGBoost": {"subsample": 0.85, "reg_lambda": 5, "min_child_weight": 1, "max_depth": 8,
                "learning_rate": 0.03, "colsample_bytree": 1.0},
}


def build_models(params: dict) -> dict:
    """Instantiate the four base models with the given parameters plus the stacked ensemble."""
    models = {n: clone(est).set_params(**params[n]) for n, (est, _) in BASE_MODELS.items()}
    models["Stacked ensemble"] = StackingClassifier(
        estimators=[(n.split()[0].lower(), clone(m)) for n, m in models.items()],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), stack_method="predict_proba", n_jobs=1)
    return models


def best_threshold(y, p) -> float:
    """Probability cut-off that maximises macro-F1 on a validation set."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])
