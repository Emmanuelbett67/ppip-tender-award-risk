"""Batch scoring from the command line (same code path as the API).

    python scripts/predict.py data/examples/draft_tenders.csv --out predictions.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from tender_risk.config import BUNDLE_PATH
from tender_risk.scoring import load_bundle, score_tenders


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="CSV with one draft tender per row (see data/examples/draft_tenders.csv)")
    ap.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    ap.add_argument("--out", type=Path, default=None, help="write predictions to this CSV (default: print JSON)")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    results = score_tenders(raw, load_bundle(args.bundle))
    if args.out:
        rows = []
        for r in results:
            r = dict(r)
            r["leading_features"] = "; ".join(f"{f['feature']} ({f['contribution']:+.2f})" for f in r.pop("leading_features", []))
            rows.append(r)
        pd.concat([raw.reset_index(drop=True), pd.DataFrame(rows)], axis=1).to_csv(args.out, index=False)
        print(f"Wrote {len(rows)} predictions to {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
