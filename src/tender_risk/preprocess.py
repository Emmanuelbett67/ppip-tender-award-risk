"""Preprocessing fitted inside every training window (no information from the future leaks in)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CAT, NUM, SEED, TEXT


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputation within a grouping column (procuring-entity type), with missingness indicators."""

    def __init__(self, group_col="pe_type"):
        self.group_col = group_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.num_cols_ = [c for c in X.columns if c != self.group_col]
        self.global_ = X[self.num_cols_].median()
        self.group_ = X.groupby(self.group_col)[self.num_cols_].median()
        self.ind_cols_ = [c for c in self.num_cols_ if X[c].isna().any()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = X[self.num_cols_].astype(float)
        ind = out[self.ind_cols_].isna().astype(float).add_suffix("_missing")
        grp = self.group_.reindex(X[self.group_col].values)
        grp.index = X.index
        out = out.fillna(grp).fillna(self.global_)
        return pd.concat([out, ind], axis=1).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.num_cols_ + [f"{c}_missing" for c in self.ind_cols_])


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", GroupMedianImputer("pe_type")), ("scale", StandardScaler())]), NUM + ["pe_type"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), CAT),
        ("txt", Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000,
                                                    sublinear_tf=True, stop_words="english")),
                          ("svd", TruncatedSVD(n_components=50, random_state=SEED))]), TEXT),
    ], sparse_threshold=0)


def base_feature(col: str) -> str:
    """Map a transformed column name back to its source feature (one-hot levels and text components are grouped)."""
    grp, name = col.split("__", 1)
    if grp == "txt":
        return "title text"
    if grp == "num":
        return name.replace("_missing", "")
    for c in sorted(CAT, key=len, reverse=True):
        if name.startswith(c + "_"):
            return c
    return name
