"""Stage 3 of the pipeline: fit the deployed model on 2018–2024 and serialize the scoring bundle.

The bundle holds the fitted preprocessor, the model, its classification threshold (chosen on the 2020 validation
year, never on test data), the risk-band cut-offs and the feature lists, so the API needs nothing else.

    python scripts/train.py                              # gradient boosting (selected in evaluate.py)
    python scripts/train.py --model XGBoost --out models/award_risk_xgb
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tender_risk
from tender_risk.explain import shap_contributions
from tender_risk.config import (CAT, DEFS, FEATURE_TABLE, MODEL_DIR, NUM, PRIMARY_DEF, REPORT_DIR, ROOT, TEXT,
                                TUNE_TRAIN_YEARS, TUNE_VAL_YEAR, X_COLS, YEARS)
from tender_risk.data import label, load_feature_table
from tender_risk.models import DEFAULT_PARAMS, MODEL_NAMES, best_threshold, build_models
from tender_risk.preprocess import make_preprocessor
from tender_risk.scoring import score_tenders


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=FEATURE_TABLE)
    ap.add_argument("--model", default="Gradient boosting", choices=MODEL_NAMES)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    ap.add_argument("--params", type=Path, default=REPORT_DIR / "metrics" / "tuning.json",
                    help="hyperparameters written by evaluate.py --tune (defaults are used if absent)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    params = DEFAULT_PARAMS
    if args.params.exists():
        params = {**DEFAULT_PARAMS, **json.loads(args.params.read_text())}
        print(f"Hyperparameters from {args.params}")

    feat = load_feature_table(args.features)
    keep, y = label(feat, PRIMARY_DEF)

    # threshold on the 2020 validation year (fit on 2018–2019), exactly as in the evaluation
    tr = feat[keep & feat["year"].isin(TUNE_TRAIN_YEARS)]
    va = feat[keep & (feat["year"] == TUNE_VAL_YEAR)]
    pre_v = make_preprocessor().fit(tr[X_COLS])
    m_v = build_models(params)[args.model].fit(pre_v.transform(tr[X_COLS]), y[tr.index])
    threshold = best_threshold(y[va.index], m_v.predict_proba(pre_v.transform(va[X_COLS]))[:, 1])
    print(f"Threshold (macro-F1 on {TUNE_VAL_YEAR}): {threshold:.2f}")

    # final fit on every tender under the primary definition
    train_all = feat[keep]
    pre = make_preprocessor().fit(train_all[X_COLS])
    X = pre.transform(train_all[X_COLS])
    model = build_models(params)[args.model].fit(X, y[train_all.index])
    risk_train = 1 - model.predict_proba(X)[:, 1]
    bundle = {
        "preprocessor": pre, "model": model, "model_name": args.model, "threshold": threshold,
        "risk_band_cutoffs": {"High": float(np.quantile(risk_train, 0.8)), "Medium": float(np.quantile(risk_train, 0.5))},
        "feature_columns": X_COLS, "numeric": NUM, "categorical": CAT, "text": TEXT,
        "outcome_definition": DEFS[PRIMARY_DEF]["name"], "trained_on": f"{YEARS[0]}-{YEARS[1]}",
        "n_train": int(len(train_all)), "created": datetime.now().isoformat(timespec="seconds"),
        "versions": {"tender_risk": tender_risk.__version__, "scikit-learn": sklearn.__version__},
    }
    joblib.dump(bundle, args.out / "award_risk_bundle.joblib")
    meta = {k: v for k, v in bundle.items() if k not in ("preprocessor", "model")}
    meta["hyperparameters"] = params.get(args.model, "stacked")
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Saved {args.out / 'award_risk_bundle.joblib'} ({len(train_all):,} training tenders)")

    # smoke test: score the example draft tenders through the same path the API uses
    loaded = joblib.load(args.out / "award_risk_bundle.joblib")
    demo = pd.read_csv(ROOT / "data" / "examples" / "draft_tenders.csv")
    for title, rec in zip(demo["title"], score_tenders(demo, loaded)):
        print(f"  {title[:45]:<45} p={rec['award_probability']:.3f} band={rec['risk_band']}")

    # global explanation of the deployed model: mean |SHAP| on a sample of the latest year
    sample = feat[keep & (feat["year"] == YEARS[1])].sample(min(5000, int((keep & (feat["year"] == YEARS[1])).sum())),
                                                            random_state=0)
    contrib = shap_contributions(loaded, pre.transform(sample[X_COLS]))
    imp = contrib.abs().mean().sort_values()
    imp.rename("mean_abs_shap").to_csv(REPORT_DIR / "metrics" / "shap_importance_deployed.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="#002D62")
    ax.set(xlabel="Mean |SHAP value| (log-odds of award)", title=f"Deployed {args.model}: feature importance, {YEARS[1]} tenders")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "figures" / "12_shap_importance_deployed.png", dpi=200, bbox_inches="tight")
    print("Top features:", ", ".join(imp.index[::-1][:5]))


if __name__ == "__main__":
    main()
