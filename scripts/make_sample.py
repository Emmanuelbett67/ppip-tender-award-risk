"""Build the small, committed sample of the raw extracts (data/sample) from the full extracts (data/raw).

A random fraction of tenders is kept; child tables are filtered to those tenders; only the columns the pipeline
uses are retained, so no bidder or supplier names, contact details or PINs are included.

    python scripts/make_sample.py --raw data/raw --out data/sample --frac 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = {
    "tenders": ["id", "pe_id", "procurement_method_id", "procurement_category_id", "title", "published_at", "close_at",
                "tender_fee", "published", "terminated", "is_agpo", "is_reservation", "description", "validity_in_days",
                "deleted_at"],
    "contracts": ["id", "tender_id", "award_date", "created_at", "deleted_at"],
    "bidders": ["id", "tender_id", "created_at", "deleted_at"],
    "tender_terminations": ["id", "tender_id", "termination_reason_id", "termination_date"],
    "addenda": ["id", "tender_id", "addendum_date", "new_closing_date", "extending_date", "deleted_at"],
    "pe": ["id", "pe_type_id", "county_id"],
}
LOOKUPS = ["pe_types", "counties", "procurement_methods", "procurement_categories", "termination_reasons"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tenders = pd.read_csv(args.raw / "tenders.csv", na_values=["NULL"], low_memory=False, usecols=KEEP["tenders"])
    rng = np.random.default_rng(args.seed)
    keep_ids = tenders.loc[rng.random(len(tenders)) < args.frac, "id"]
    sample = tenders[tenders["id"].isin(keep_ids)].copy()
    sample["description"] = sample["description"].astype("string").str.slice(0, 120)
    sample.to_csv(args.out / "tenders.csv", index=False)
    print(f"tenders: {len(sample):,} of {len(tenders):,}")

    for t in ["contracts", "bidders", "tender_terminations", "addenda"]:
        df = pd.read_csv(args.raw / f"{t}.csv", na_values=["NULL"], low_memory=False, usecols=KEEP[t])
        df = df[df["tender_id"].isin(keep_ids)]
        df.to_csv(args.out / f"{t}.csv", index=False)
        print(f"{t}: {len(df):,}")

    pe = pd.read_csv(args.raw / "pe.csv", na_values=["NULL"], low_memory=False, usecols=KEEP["pe"])
    pe.to_csv(args.out / "pe.csv", index=False)
    for t in LOOKUPS:
        pd.read_csv(args.raw / f"{t}.csv", na_values=["NULL"], low_memory=False).to_csv(args.out / f"{t}.csv", index=False)
    print("lookup tables copied:", ", ".join(["pe"] + LOOKUPS))


if __name__ == "__main__":
    main()
