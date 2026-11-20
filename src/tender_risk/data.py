"""Data pipeline: portal extracts -> analysis population -> outcome classes -> feature table.

Mirrors Sections 3.1–3.4 of the research notebook, as a reusable function.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, DEFS, META_COLS, STRUCTURAL_NO_CONTRACT, X_COLS, YEARS
from .features import engineer_features

TABLES = ["tenders", "contracts", "bidders", "tender_terminations", "addenda", "pe", "pe_types",
          "counties", "procurement_methods", "procurement_categories"]


def read_table(raw_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / f"{name}.csv", na_values=["NULL"], low_memory=False)


def load_extracts(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the PPIP CSV extracts and parse the tender dates."""
    missing = [t for t in TABLES if not (raw_dir / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing extracts in {raw_dir}: {missing}")
    db = {t: read_table(raw_dir, t) for t in TABLES}
    db["bidders"] = db["bidders"][db["bidders"]["deleted_at"].isna()]   # soft-deleted bid records do not count as activity
    ten = db["tenders"]
    ten["published_at"] = pd.to_datetime(ten["published_at"], errors="coerce")
    ten["close_at"] = pd.to_datetime(ten["close_at"], errors="coerce")
    ten["year"] = ten["published_at"].dt.year
    return db


def build_population(db: dict[str, pd.DataFrame], sample_frac: float | None = None, seed: int = 42):
    """Apply the inclusion criteria. Returns (population, flow table)."""
    tenders, pe, pe_types, counties, methods = (db[t] for t in ["tenders", "pe", "pe_types", "counties", "procurement_methods"])
    pe_info = (pe[["id", "pe_type_id", "county_id"]]
               .merge(pe_types[["id", "description"]].rename(columns={"id": "pe_type_id", "description": "pe_type"}), how="left")
               .merge(counties[["id", "name"]].rename(columns={"id": "county_id", "name": "county"}), how="left")
               .rename(columns={"id": "pe_id"}))
    method_name = methods.set_index("id")["title"]

    flow = [("All tender records", len(tenders))]
    pop = tenders[tenders["deleted_at"].isna()]
    flow.append(("Not soft-deleted", len(pop)))
    pop = pop[pop["year"].between(*YEARS)]
    flow.append((f"Published {YEARS[0]}–{YEARS[1]}", len(pop)))
    pop = pop[pop["published"] == 1]
    flow.append(("Flagged as published on the portal", len(pop)))
    pop = pop.assign(procurement_method=pop["procurement_method_id"].map(method_name))
    pop = pop[~pop["procurement_method"].isin(STRUCTURAL_NO_CONTRACT)]
    flow.append(("Excluding EOI, prequalification and disposal (no contract by design)", len(pop)))
    pop = pop.merge(pe_info, on="pe_id", how="left").reset_index(drop=True)
    if sample_frac:
        pop = pop.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        flow.append((f"Subsample ({sample_frac:.0%})", len(pop)))
    return pop, pd.DataFrame(flow, columns=["Step", "Tenders"])


def assign_outcome_class(pop: pd.DataFrame, db: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each tender gets exactly one class, in priority order: Awarded > Terminated > Undetermined > No activity."""
    c_ = pop["id"].isin(db["contracts"]["tender_id"])
    t_ = pop["id"].isin(db["tender_terminations"]["tender_id"])
    b_ = pop["id"].isin(db["bidders"]["tender_id"])
    return pop.assign(outcome_class=np.select([c_, t_, b_], CLASSES[:3], CLASSES[3]))


def attach_pre_close_inputs(pop: pd.DataFrame, db: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add the raw inputs that need other tables: addenda before closing and entity experience."""
    tenders, addenda, categories = db["tenders"], db["addenda"], db["procurement_categories"]

    ad = addenda[addenda["deleted_at"].isna()].merge(tenders[["id", "close_at"]], left_on="tender_id", right_on="id",
                                                     suffixes=("", "_tender"))
    ad["addendum_date"] = pd.to_datetime(ad["addendum_date"], errors="coerce")
    ad = ad[ad["addendum_date"] <= ad["close_at"]]
    ad["extends"] = ad["new_closing_date"].notna() | ad["extending_date"].notna()
    ad_agg = ad.groupby("tender_id").agg(n_addenda_pre_close=("id", "size"), closing_extended=("extends", "max"))
    out = pop.merge(ad_agg, left_on="id", right_index=True, how="left")
    out["n_addenda_pre_close"] = out["n_addenda_pre_close"].fillna(0)
    out["closing_extended"] = out["closing_extended"].astype("boolean").fillna(False).astype(int)

    allpub = tenders.loc[tenders["deleted_at"].isna(), ["id", "pe_id", "published_at"]].sort_values(["published_at", "id"])
    allpub["pe_prior_tenders"] = allpub.groupby("pe_id").cumcount()
    out = out.merge(allpub[["id", "pe_prior_tenders"]], on="id", how="left")

    out["procurement_category"] = out["procurement_category_id"].map(categories.set_index("id")["title"])
    out["agpo"] = out["is_agpo"].map({1.0: "AGPO", 0.0: "Non-AGPO"}).fillna("Unknown")
    return out


def build_feature_table(raw_dir: Path, sample_frac: float | None = None, seed: int = 42):
    """Full pipeline. Returns (feature table, inclusion flow)."""
    db = load_extracts(raw_dir)
    pop, flow = build_population(db, sample_frac, seed)
    pop = assign_outcome_class(pop, db)
    pop = attach_pre_close_inputs(pop, db)
    feat = engineer_features(pop)
    return feat[META_COLS + X_COLS].copy(), flow


def label(df: pd.DataFrame, definition: str):
    """Rows kept under an outcome definition and the binary label (1 = awarded)."""
    keep = df["outcome_class"].isin(["Awarded"] + DEFS[definition]["neg"])
    return keep, (df["outcome_class"] == "Awarded").astype(int)


def load_feature_table(path: Path) -> pd.DataFrame:
    feat = pd.read_parquet(path)
    for c in ["procurement_method", "procurement_category", "pe_type", "county", "pub_month", "fiscal_quarter"]:
        feat[c] = feat[c].astype(object)
    return feat
