import numpy as np
import pandas as pd

from tender_risk.config import X_COLS
from tender_risk.features import engineer_features, fiscal_quarter


def test_fiscal_quarter_starts_in_july():
    months = pd.Series([7, 9, 10, 12, 1, 3, 4, 6])
    assert fiscal_quarter(months).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_engineer_features_produces_every_predictor():
    raw = pd.DataFrame([{
        "title": "Supply of stationery", "description": "x", "procurement_method": "Open Tender",
        "procurement_category": "Goods", "pe_type": "Ministry", "county": "Nairobi",
        "published_at": "2024-03-04", "close_at": "2024-03-25", "validity_in_days": 120, "tender_fee": 1000,
        "n_addenda_pre_close": 1, "closing_extended": 1, "pe_prior_tenders": 10,
    }])
    f = engineer_features(raw)
    assert set(X_COLS) <= set(f.columns)
    row = f.iloc[0]
    assert row["days_to_close"] == 21
    assert row["pub_month"] == "3" and row["fiscal_quarter"] == "Q3"
    assert row["days_to_fy_end"] == (pd.Timestamp("2024-06-30") - pd.Timestamp("2024-03-04")).days
    assert row["has_tender_fee"] == 1 and np.isclose(row["log_tender_fee"], np.log1p(1000))
    assert row["title_words"] == 3 and row["has_description"] == 1
    assert np.isclose(row["log_pe_prior_tenders"], np.log1p(10))


def test_out_of_range_values_become_missing_not_errors():
    raw = pd.DataFrame([{"title": "", "published_at": "2024-01-01", "close_at": "2025-06-01",
                         "validity_in_days": 5000, "tender_fee": None}])
    f = engineer_features(raw)
    row = f.iloc[0]
    assert np.isnan(row["days_to_close"])        # > 180 days
    assert np.isnan(row["validity_days"])        # > 730 days
    assert row["has_tender_fee"] == 0 and row["log_tender_fee"] == 0
    assert row["has_description"] == 0
    assert (f[["procurement_method", "county"]] == "Unknown").all().all()
