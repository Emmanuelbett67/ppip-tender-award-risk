"""Stage 2 of the pipeline: forward-chaining temporal validation of the five model families.

Reproduces Sections 4–7 of the research notebook from the processed feature table and writes
metrics, significance tests, fairness tables and figures to reports/.

    python scripts/evaluate.py                  # full run (~40 min on 8 cores)
    python scripts/evaluate.py --fast           # 15% subsample, fewer bootstrap draws (~5 min)
    python scripts/evaluate.py --tune           # re-run the randomised hyperparameter search first
    python scripts/evaluate.py --definitions D2 --models "Gradient boosting" "XGBoost"
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, chi2_contingency, norm, rankdata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from statsmodels.stats.contingency_tables import mcnemar

from tender_risk.config import (DEFS, FEATURE_TABLE, N_BOOT, N_ITER, PRIMARY_DEF, REPORT_DIR, SEED, TEST_YEARS,
                                TUNE_TRAIN_YEARS, TUNE_VAL_YEAR, X_COLS, YEARS)
from tender_risk.data import label, load_feature_table
from tender_risk.models import BASE_MODELS, DEFAULT_PARAMS, MODEL_NAMES, best_threshold, build_models
from tender_risk.preprocess import make_preprocessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAVY, RED, GREY = "#002D62", "#C8102E", "#9AA5B1"


def metrics(y, p, yhat):
    return {"Accuracy": accuracy_score(y, yhat), "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat), "F1 (award)": f1_score(y, yhat),
            "F1 (non-award)": f1_score(1 - y, 1 - yhat), "Balanced acc.": balanced_accuracy_score(y, yhat),
            "ROC-AUC": roc_auc_score(y, p)}


def delong(y, p1, p2):
    """DeLong test for two correlated ROC curves (fast algorithm of Sun & Xu, 2014)."""
    y = np.asarray(y).astype(bool)
    m, n_ = y.sum(), (~y).sum()
    aucs, v10, v01 = [], [], []
    for p in (np.asarray(p1), np.asarray(p2)):
        x, z = p[y], p[~y]
        tx, tz, tall = rankdata(x), rankdata(z), rankdata(np.concatenate([x, z]))
        aucs.append((tall[:m].sum() - m * (m + 1) / 2) / (m * n_))
        v10.append((tall[:m] - tx) / n_)
        v01.append(1 - (tall[m:] - tz) / m)
    S = np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n_
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return diff, z, 2 * norm.sf(abs(z))


def ece(y, p, bins=10):
    b = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "b": b}).groupby("b")
    return float((d.size() / len(b) * (d["y"].mean() - d["p"].mean()).abs()).sum())


def tune(feat, n_iter):
    keep, y_all = label(feat, PRIMARY_DEF)
    t = feat[keep & feat["year"].isin(TUNE_TRAIN_YEARS + [TUNE_VAL_YEAR])]
    is_val = (t["year"] == TUNE_VAL_YEAR).to_numpy()
    pre = make_preprocessor().fit(t.loc[~is_val, X_COLS])
    X = pre.transform(t[X_COLS])
    split = PredefinedSplit(np.where(is_val, 0, -1))
    params, rows = {}, []
    for name, (est, space) in BASE_MODELS.items():   # always tune all four: the stack reuses them
        search = RandomizedSearchCV(est, space, n_iter=min(n_iter, int(np.prod([len(v) for v in space.values()]))),
                                    scoring="roc_auc", cv=split, random_state=SEED, refit=False, n_jobs=1)
        search.fit(X, y_all[t.index])
        params[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        rows.append({"model": name, "validation ROC-AUC (2020)": round(search.best_score_, 4), "best parameters": params[name]})
        print(f"  tuned {name}: AUC(2020) = {search.best_score_:.4f} {params[name]}")
    return params, pd.DataFrame(rows)


def thresholds_on_validation(feat, params, defs, models):
    thr = {}
    for d in defs:
        keep_d, y_d = label(feat, d)
        tr = feat[keep_d & feat["year"].isin(TUNE_TRAIN_YEARS)]
        va = feat[keep_d & (feat["year"] == TUNE_VAL_YEAR)]
        pre = make_preprocessor().fit(tr[X_COLS])
        Xtr, Xva = pre.transform(tr[X_COLS]), pre.transform(va[X_COLS])
        for name, m in build_models(params).items():
            if name not in models:
                continue
            m.fit(Xtr, y_d[tr.index])
            thr[(d, name)] = best_threshold(y_d[va.index], m.predict_proba(Xva)[:, 1])
    return thr


def forward_chain(feat, params, thr, defs, models):
    t0, rows = time.time(), []
    for d in defs:
        keep_d, y_d = label(feat, d)
        for Y in TEST_YEARS:
            tr = feat[keep_d & feat["year"].between(YEARS[0], Y - 1)]
            te = feat[keep_d & (feat["year"] == Y)]
            pre = make_preprocessor().fit(tr[X_COLS])
            Xtr, Xte = pre.transform(tr[X_COLS]), pre.transform(te[X_COLS])
            for name, m in build_models(params).items():
                if name not in models:
                    continue
                m.fit(Xtr, y_d[tr.index])
                rows.append(pd.DataFrame({"definition": d, "test_year": Y, "model": name, "id": te["id"].values,
                                          "y": y_d[te.index].values, "p": m.predict_proba(Xte)[:, 1]}))
            print(f"  {d} | test {Y} | train n={len(tr):,} | test n={len(te):,} | {(time.time() - t0) / 60:.1f} min")
    preds = pd.concat(rows, ignore_index=True)
    preds["thr"] = [thr[(d, m)] for d, m in zip(preds["definition"], preds["model"])]
    preds["yhat"] = (preds["p"] >= preds["thr"]).astype(int)
    return preds


def fairness_tables(preds, feat, best, out_dir, n_boot):
    F = (preds[(preds["definition"] == PRIMARY_DEF) & (preds["model"] == best)]
         .merge(feat[["id", "agpo", "county"]], on="id"))
    F["flag"] = 1 - F["yhat"]

    def group_table(df, by):
        g = df.groupby(by)
        return pd.DataFrame({
            "tenders": g.size(), "base rate (award)": g["y"].mean(), "flag rate": g["flag"].mean(),
            "false-flag rate (awarded but flagged)": g.apply(lambda s: s.loc[s["y"] == 1, "flag"].mean()),
            "FPR (not awarded, predicted award)": g.apply(lambda s: s.loc[s["y"] == 0, "yhat"].mean()),
            "mean score": g["p"].mean(), "ECE": g.apply(lambda s: ece(s["y"], s["p"]))})

    FA = F[F["agpo"] != "Unknown"]
    agpo_tab = group_table(FA, "agpo")
    agpo_tab.round(4).to_csv(out_dir / "fairness_by_agpo.csv")

    crit = {"Independence: flag rate": lambda s: s["flag"].mean(),
            "Separation: false-flag rate (Y=1)": lambda s: s.loc[s["y"] == 1, "flag"].mean(),
            "Separation: FPR (Y=0)": lambda s: s.loc[s["y"] == 0, "yhat"].mean(),
            "Sufficiency: ECE": lambda s: ece(s["y"], s["p"])}
    a, b = FA[FA["agpo"] == "AGPO"], FA[FA["agpo"] == "Non-AGPO"]
    r = np.random.default_rng(SEED)
    rows = []
    for k, f in crit.items():
        point = f(a) - f(b)
        draws = [f(a.iloc[r.integers(0, len(a), len(a))]) - f(b.iloc[r.integers(0, len(b), len(b))]) for _ in range(n_boot)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"criterion (AGPO - Non-AGPO)": k, "difference": round(point, 4), "CI low": round(lo, 4),
                     "CI high": round(hi, 4), "CI excludes 0": not (lo <= 0 <= hi)})
    chi_p = chi2_contingency(pd.crosstab(FA["agpo"], FA["flag"]))[1]
    lg = np.log(np.clip(FA["p"], 1e-6, 1 - 1e-6) / (1 - np.clip(FA["p"], 1e-6, 1 - 1e-6)))
    a_ = (FA["agpo"] == "AGPO").astype(float)
    base_m = sm.Logit(FA["y"], sm.add_constant(pd.DataFrame({"logit": lg}))).fit(disp=0)
    full_m = sm.Logit(FA["y"], sm.add_constant(pd.DataFrame({"logit": lg, "agpo": a_, "logit_x_agpo": lg * a_}))).fit(disp=0)
    lr_stat = 2 * (full_m.llf - base_m.llf)
    rows.append({"criterion (AGPO - Non-AGPO)": "Independence chi-square p", "difference": chi_p})
    rows.append({"criterion (AGPO - Non-AGPO)": "Sufficiency LR-test p (df=2)", "difference": chi2.sf(lr_stat, 2)})
    pd.DataFrame(rows).to_csv(out_dir / "fairness_agpo_tests.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cal = FA.assign(bin=np.minimum((FA["p"] * 10).astype(int), 9))
    for grp, col in [("AGPO", RED), ("Non-AGPO", NAVY)]:
        c = cal[cal["agpo"] == grp].groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        c = c[c["n"] >= 30]
        axes[0].plot(c["p"], c["y"], marker="o", color=col, label=grp)
    axes[0].plot([0, 1], [0, 1], ls="--", color=GREY)
    axes[0].set(xlabel="Mean predicted award probability", ylabel="Observed award rate", title="Sufficiency: calibration by AGPO status")
    axes[0].legend()
    bars = agpo_tab[["flag rate", "false-flag rate (awarded but flagged)", "FPR (not awarded, predicted award)"]].T
    bars.plot(kind="bar", ax=axes[1], color=[RED, NAVY], rot=0)
    axes[1].set(title="Independence and separation by AGPO status", ylabel="Rate")
    axes[1].set_xticklabels(["Flag rate", "False-flag rate\n(awarded)", "FPR\n(not awarded)"])
    plt.tight_layout()
    plt.savefig(out_dir.parent / "figures" / "10_fairness_agpo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ct = group_table(F, "county")
    ct = ct[ct["tenders"] >= 300].sort_values("flag rate")
    ct.round(4).to_csv(out_dir / "fairness_by_county.csv")
    return agpo_tab, pd.DataFrame(rows), ct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=FEATURE_TABLE)
    ap.add_argument("--out", type=Path, default=REPORT_DIR)
    ap.add_argument("--tune", action="store_true", help="re-run the randomised hyperparameter search")
    ap.add_argument("--fast", action="store_true", help="15%% subsample and 200 bootstrap draws (smoke test)")
    ap.add_argument("--definitions", nargs="+", default=list(DEFS), choices=list(DEFS))
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    args = ap.parse_args()

    metrics_dir, fig_dir = args.out / "metrics", args.out / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 200 if args.fast else N_BOOT
    defs = [d for d in DEFS if d in args.definitions]
    if PRIMARY_DEF not in defs:
        raise SystemExit(f"--definitions must include the primary definition {PRIMARY_DEF}")
    models = [m for m in MODEL_NAMES if m in args.models]

    feat = load_feature_table(args.features)
    if args.fast:
        feat = feat.sample(frac=0.15, random_state=SEED).reset_index(drop=True)
    print(f"Feature table: {len(feat):,} tenders | definitions {defs} | models {models}")

    # 1. hyperparameters
    tuning_file = metrics_dir / "tuning.json"
    if args.tune:
        print("Tuning hyperparameters on 2018-2019 -> 2020 ...")
        params, tuning_tab = tune(feat, N_ITER)
        params = {**DEFAULT_PARAMS, **params}
        tuning_file.write_text(json.dumps(params, indent=2))
        tuning_tab.to_csv(metrics_dir / "tuning_results.csv", index=False)
    elif tuning_file.exists():
        params = {**DEFAULT_PARAMS, **json.loads(tuning_file.read_text())}
        print(f"Using hyperparameters from {tuning_file}")
    else:
        params = DEFAULT_PARAMS
        print("Using default hyperparameters (tender_risk.models.DEFAULT_PARAMS)")

    # 2. thresholds and forward-chaining predictions
    print("Selecting classification thresholds on the 2020 validation year ...")
    thr = thresholds_on_validation(feat, params, defs, models)
    pd.Series(thr).unstack().round(2).to_csv(metrics_dir / "thresholds.csv")
    print("Forward-chaining temporal validation ...")
    preds = forward_chain(feat, params, thr, defs, models)
    preds.to_csv(metrics_dir / "out_of_time_predictions.csv", index=False)

    # 3. metrics
    per_year = (preds.groupby(["definition", "model", "test_year"])
                .apply(lambda g: pd.Series(metrics(g["y"], g["p"], g["yhat"]))).reset_index())
    pooled = (preds.groupby(["definition", "model"])
              .apply(lambda g: pd.Series(metrics(g["y"], g["p"], g["yhat"]))).reset_index())
    pooled.to_csv(metrics_dir / "metrics_pooled.csv", index=False)
    per_year.to_csv(metrics_dir / "metrics_by_year.csv", index=False)
    prim = pooled[pooled["definition"] == PRIMARY_DEF].set_index("model")
    print("\nPooled out-of-time metrics, primary definition:")
    print(prim.round(4).to_string())

    auc_year = per_year[per_year["definition"] == PRIMARY_DEF].pivot(index="model", columns="test_year", values="ROC-AUC")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (m, row) in enumerate(auc_year.iterrows()):
        ax.plot(row.index, row.values, marker="o", label=m)
    ax.set(xlabel="Test year", ylabel="ROC-AUC", title=f"ROC-AUC by test year ({DEFS[PRIMARY_DEF]['name']})", xticks=TEST_YEARS)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "06_auc_by_year.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. bootstrap confidence intervals (primary definition)
    rng = np.random.default_rng(SEED)
    P = preds[preds["definition"] == PRIMARY_DEF]
    n = P.groupby("model").size().iloc[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    ci_rows = []
    for name, g in P.groupby("model", sort=False):
        y, p, yh = g["y"].to_numpy(), g["p"].to_numpy(), g["yhat"].to_numpy()
        boot = np.array([[roc_auc_score(y[i], p[i]), f1_score(1 - y[i], 1 - yh[i]), accuracy_score(y[i], yh[i])] for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        point = [roc_auc_score(y, p), f1_score(1 - y, 1 - yh), accuracy_score(y, yh)]
        ci_rows.append({"model": name, **{k: f"{v:.3f} [{a:.3f}, {b:.3f}]" for k, v, a, b in
                                          zip(["ROC-AUC", "F1 (non-award)", "Accuracy"], point, lo, hi)}})
    ci = pd.DataFrame(ci_rows).set_index("model")
    ci.to_csv(metrics_dir / "bootstrap_ci.csv")
    print("\nBootstrap 95% CIs:")
    print(ci.to_string())

    # 5. model selection and significance tests
    best = prim.sort_values("ROC-AUC", ascending=False).index[0]
    wide = P.pivot(index="id", columns="model", values="p")
    wide_hat = P.pivot(index="id", columns="model", values="yhat")
    y_w = P.drop_duplicates("id").set_index("id")["y"].reindex(wide.index)
    tests = []
    for other in models:
        if other == best:
            continue
        diff, z, p_dl = delong(y_w, wide[best], wide[other])
        correct_b, correct_o = wide_hat[best] == y_w, wide_hat[other] == y_w
        table = pd.crosstab(correct_b, correct_o).reindex(index=[True, False], columns=[True, False], fill_value=0)
        mc = mcnemar(table.values, exact=False, correction=True)
        tests.append({"comparison": f"{best} vs {other}", "AUC difference": round(diff, 4), "DeLong z": round(z, 2),
                      "DeLong p": p_dl, "McNemar chi2": round(mc.statistic, 1), "McNemar p": mc.pvalue})
    tests = pd.DataFrame(tests)
    tests.to_csv(metrics_dir / "significance_tests.csv", index=False)
    print(f"\nSelected model (highest pooled ROC-AUC): {best}")
    print(tests.to_string(index=False))

    # 6. sensitivity to the outcome definition
    if len(defs) > 1:
        sens = pooled.pivot(index="model", columns="definition", values="ROC-AUC").reindex(models)
        sens.columns = [DEFS[c]["name"] for c in sens.columns]
        ax = sens.plot(kind="bar", figsize=(9, 4), rot=0)
        ax.set(ylabel="Pooled out-of-time ROC-AUC", title="Sensitivity of performance to the outcome definition",
               ylim=(max(0.5, sens.values.min() - 0.05), 1))
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "07_auc_by_definition.png", dpi=200, bbox_inches="tight")
        plt.close()

    # 7. fairness audit of the selected model
    print("\nFairness audit (AGPO status and county) ...")
    agpo_tab, fair, ct = fairness_tables(preds, feat, best, metrics_dir, n_boot)
    print(agpo_tab.round(4).to_string())
    print(fair.to_string(index=False))
    print(f"{len(ct)} counties audited; flag-rate range {ct['flag rate'].min():.3f}-{ct['flag rate'].max():.3f}")

    summary = {"selected_model": best, "primary_definition": DEFS[PRIMARY_DEF]["name"], "n_tenders": int(len(feat)),
               "pooled_auc": float(prim.loc[best, "ROC-AUC"]), "pooled_f1_non_award": float(prim.loc[best, "F1 (non-award)"]),
               "fast": args.fast, "definitions": defs, "models": models}
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone. Outputs in", args.out)


if __name__ == "__main__":
    main()
