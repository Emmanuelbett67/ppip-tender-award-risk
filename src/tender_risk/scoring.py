"""Load the serialized bundle and score draft tenders. Shared by the API and the batch CLI."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import BUNDLE_PATH
from .explain import leading_features, shap_contributions
from .features import engineer_features


def load_bundle(path: Path | None = None) -> dict:
    return joblib.load(path or BUNDLE_PATH)


def risk_band(risk: np.ndarray, cutoffs: dict) -> np.ndarray:
    return np.where(risk >= cutoffs["High"], "High", np.where(risk >= cutoffs["Medium"], "Medium", "Low"))


def score_tenders(raw: pd.DataFrame, bundle: dict, top_k: int = 3, explain: bool = True) -> list[dict]:
    """Score raw tender records (same columns the API accepts). Returns one dict per tender."""
    feat = engineer_features(raw)
    X = bundle["preprocessor"].transform(feat[bundle["feature_columns"]])
    p = bundle["model"].predict_proba(X)[:, 1]
    risk = 1 - p
    bands = risk_band(risk, bundle["risk_band_cutoffs"])
    contrib = shap_contributions(bundle, X) if explain else None
    results = []
    for i in range(len(feat)):
        rec = {
            "award_probability": round(float(p[i]), 4),
            "failure_risk": round(float(risk[i]), 4),
            "risk_band": str(bands[i]),
            "flagged_at_risk": bool(p[i] < bundle["threshold"]),
        }
        if contrib is not None:
            rec["leading_features"] = leading_features(contrib.iloc[i], top_k)
        results.append(rec)
    return results
