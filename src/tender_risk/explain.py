"""Per-tender explanations: which features pushed the award probability up or down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .preprocess import base_feature


def shap_contributions(bundle: dict, X: np.ndarray) -> pd.DataFrame:
    """TreeSHAP contributions (log-odds of award) summed back to the original feature names.

    Works for scikit-learn HistGradientBoosting and XGBoost models alike through `shap.TreeExplainer`.
    """
    explainer = bundle.get("_explainer")
    if explainer is None:
        explainer = shap.TreeExplainer(bundle["model"])
        bundle["_explainer"] = explainer
    values = explainer.shap_values(X)
    if isinstance(values, list):          # some versions return one array per class
        values = values[1]
    if values.ndim == 3:                   # (n, features, classes)
        values = values[:, :, -1]
    cols = bundle["preprocessor"].get_feature_names_out()
    groups = [base_feature(c) for c in cols]
    return pd.DataFrame(values, columns=cols).T.groupby(groups).sum().T


def leading_features(contrib_row: pd.Series, top_k: int = 3) -> list[dict]:
    """The `top_k` features with the largest absolute contribution, as API-friendly records."""
    top = contrib_row.sort_values(key=np.abs, ascending=False).head(top_k)
    return [{"feature": f, "contribution": round(float(v), 4),
             "direction": "raises award probability" if v > 0 else "lowers award probability"}
            for f, v in top.items()]
