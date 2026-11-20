"""Project-wide constants: analysis window, outcome definitions and feature lists.

Everything here is a fixed study-design decision taken in the proposal; nothing is tuned.
"""

from pathlib import Path

SEED = 42

# ----------------------------- paths -----------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_TABLE = PROCESSED_DIR / "tenders_features.parquet"
MODEL_DIR = ROOT / "models" / "award_risk_v2"
BUNDLE_PATH = MODEL_DIR / "award_risk_bundle.joblib"
REPORT_DIR = ROOT / "reports"

# ----------------------------- study design -----------------------------
YEARS = (2018, 2024)              # analysis window (over 99% of portal records)
TUNE_TRAIN_YEARS = [2018, 2019]   # hyperparameter tuning: fit on these ...
TUNE_VAL_YEAR = 2020              # ... validate on this (precedes every test year)
TEST_YEARS = [2021, 2022, 2023, 2024]
PRIMARY_DEF = "D2"                # pre-specified primary outcome definition
N_ITER = 10                       # randomised-search iterations per model
N_BOOT = 1000                     # bootstrap resamples for confidence intervals

# Procurement methods that never produce a contract by design.
STRUCTURAL_NO_CONTRACT = ["Expression of Interest", "Prequalification", "Disposal of assets"]

# Outcome taxonomy, in priority order.
CLASSES = ["Awarded", "Terminated", "Undetermined", "No recorded activity"]

# Alternative outcome definitions. y = 1 always means "reached contract award".
DEFS = {
    "D1": {"name": "D1 Naive", "neg": ["Terminated", "Undetermined", "No recorded activity"]},
    "D2": {"name": "D2 Observed activity", "neg": ["Terminated", "Undetermined"]},
    "D3": {"name": "D3 Strict failure", "neg": ["Terminated"]},
}

# ----------------------------- features -----------------------------
# Only information that exists by the tender's closing date is admissible.
NUM = ["days_to_close", "validity_days", "has_tender_fee", "log_tender_fee", "days_to_fy_end",
       "n_addenda_pre_close", "closing_extended", "log_pe_prior_tenders", "title_words", "has_description"]
CAT = ["procurement_method", "procurement_category", "pe_type", "county", "pub_month", "fiscal_quarter"]
TEXT = "title"
X_COLS = NUM + CAT + [TEXT]

# Columns kept in the processed feature table besides the predictors.
META_COLS = ["id", "pe_id", "year", "published_at", "close_at", "outcome_class", "agpo"]
