"""Feature engineering from tender-level fields known by the closing date.

This module is the single owner of the derived-feature definitions. The data pipeline calls it on the
whole portal extract; the API calls it on one draft tender. Keeping one implementation guarantees that
training and serving compute identical features.

Expected input columns (missing ones are treated as absent / unknown):
    published_at, close_at        datetime
    validity_in_days, tender_fee  numeric
    n_addenda_pre_close           count of addenda issued on or before the closing date
    closing_extended              0/1, closing date extended by an addendum
    pe_prior_tenders              number of tenders the entity published before this one
    title, description            text
    procurement_method, procurement_category, pe_type, county   categorical strings
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAT, NUM, TEXT


def fiscal_quarter(month: pd.Series) -> pd.Series:
    """Kenyan fiscal year runs July–June: July = Q1."""
    return ((month - 7) % 12) // 3 + 1


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column `name`, or an all-missing series when the input lacks it."""
    return df[name] if name in df else pd.Series(np.nan, index=df.index, dtype=object)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the NUM, CAT and TEXT predictor columns added."""
    out = df.copy()
    pub = pd.to_datetime(_col(out, "published_at"), errors="coerce")
    close = pd.to_datetime(_col(out, "close_at"), errors="coerce")

    # timing
    out["days_to_close"] = (close - pub).dt.days
    out.loc[~out["days_to_close"].between(0, 180), "days_to_close"] = np.nan
    out["pub_month"] = pub.dt.month.astype("Int64").astype(str).replace("<NA>", "Unknown")
    out["fiscal_quarter"] = ("Q" + fiscal_quarter(pub.dt.month).astype("Int64").astype(str)).replace("Q<NA>", "Unknown")
    fy_end_year = pub.dt.year + (pub.dt.month >= 7).astype(int)
    fy_end = pd.to_datetime(fy_end_year.astype("Int64").astype(str) + "-06-30", errors="coerce")
    out["days_to_fy_end"] = (fy_end - pub).dt.days

    # tender design
    validity = pd.to_numeric(_col(out, "validity_in_days"), errors="coerce")
    out["validity_days"] = validity.where(validity.between(1, 730))
    fee = pd.to_numeric(_col(out, "tender_fee"), errors="coerce")
    out["has_tender_fee"] = (fee.fillna(0) > 0).astype(int)
    out["log_tender_fee"] = np.log1p(fee.clip(lower=0).fillna(0))

    # changes before closing
    out["n_addenda_pre_close"] = pd.to_numeric(_col(out, "n_addenda_pre_close"), errors="coerce").fillna(0)
    out["closing_extended"] = pd.to_numeric(_col(out, "closing_extended"), errors="coerce").fillna(0).astype(int)

    # procuring-entity experience (label-free, uses the past only)
    prior = pd.to_numeric(_col(out, "pe_prior_tenders"), errors="coerce")
    out["log_pe_prior_tenders"] = np.log1p(prior.clip(lower=0))

    # text
    out[TEXT] = _col(out, TEXT).fillna("").astype(str)
    out["title_words"] = out[TEXT].str.split().str.len()
    desc = _col(out, "description")
    out["has_description"] = (desc.notna() & (desc.astype(str).str.strip() != "")).astype(int)

    # categoricals
    for c in CAT:
        if c not in out:
            out[c] = "Unknown"
        out[c] = out[c].fillna("Unknown").astype(object)

    # make sure every numeric predictor is float
    for c in NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


__all__ = ["engineer_features", "fiscal_quarter", "NUM", "CAT", "TEXT"]
