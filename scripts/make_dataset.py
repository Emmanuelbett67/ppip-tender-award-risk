"""Stage 1 of the pipeline: PPIP CSV extracts -> processed feature table (parquet).

    python scripts/make_dataset.py --raw data/raw --out data/processed/tenders_features.parquet
    python scripts/make_dataset.py --raw data/sample --out data/processed/sample_features.parquet
"""

import argparse
from pathlib import Path

from tender_risk.config import FEATURE_TABLE, RAW_DIR
from tender_risk.data import build_feature_table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="directory holding the PPIP CSV extracts")
    ap.add_argument("--out", type=Path, default=FEATURE_TABLE, help="parquet file to write")
    ap.add_argument("--sample-frac", type=float, default=None, help="optional random subsample of the population")
    args = ap.parse_args()

    feat, flow = build_feature_table(args.raw, args.sample_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(args.out, index=False)

    print("Inclusion flow:")
    print(flow.to_string(index=False))
    print("\nOutcome classes:")
    print(feat["outcome_class"].value_counts().to_string())
    print(f"\nWrote {len(feat):,} rows x {feat.shape[1]} columns to {args.out}")


if __name__ == "__main__":
    main()
