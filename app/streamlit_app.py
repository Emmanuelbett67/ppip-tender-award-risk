"""Streamlit front-end for the tender award-risk service.

A procurement officer fills in a draft tender, sees the award probability, the risk band and the reasons behind
the score, and can try changes to the tender before it is advertised. The UI calls the FastAPI service
(``TENDER_RISK_API``, default http://localhost:8000) and falls back to scoring in-process from the committed
bundle when the service is unreachable, so the demo never depends on a second process.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from tender_risk.config import BUNDLE_PATH, ROOT
from tender_risk.scoring import load_bundle, score_tenders

API_URL = os.environ.get("TENDER_RISK_API", "http://localhost:8000").rstrip("/")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BUNDLE_PATH))
REPORT_DIR = ROOT / "reports"

NAVY, GREEN, AMBER, RED, GREY = "#002D62", "#2E8B57", "#E08A1E", "#C0392B", "#8A94A6"
BAND_COLOR = {"Low": GREEN, "Medium": AMBER, "High": RED}
FEATURE_LABEL = {
    "procurement_method": "Procurement method", "procurement_category": "Procurement category",
    "pe_type": "Type of procuring entity", "county": "County of the entity",
    "days_to_close": "Length of the bidding window", "validity_days": "Tender validity period",
    "has_tender_fee": "Whether a tender fee is charged", "log_tender_fee": "Size of the tender fee",
    "days_to_fy_end": "Days left in the fiscal year", "n_addenda_pre_close": "Addenda issued",
    "closing_extended": "Closing date extended", "log_pe_prior_tenders": "Entity's prior tenders",
    "title_words": "Length of the title", "has_description": "Description provided",
    "title text": "Wording of the title", "pub_month": "Month of publication", "fiscal_quarter": "Fiscal quarter",
}

st.set_page_config(page_title="Tender Award Risk", page_icon="📋", layout="wide")
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; }}
      .band-pill {{ display:inline-block; padding:0.35rem 1rem; border-radius:999px; color:white;
                    font-weight:700; font-size:1.05rem; letter-spacing:0.02em; }}
      .reason {{ border-left: 5px solid {GREY}; padding: 0.45rem 0.8rem; margin: 0.35rem 0;
                 background: rgba(128,128,128,0.07); border-radius: 0 6px 6px 0; }}
      h1 {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------------- backends
@st.cache_resource(show_spinner="Loading the model bundle…")
def local_bundle() -> dict:
    return load_bundle(MODEL_PATH)


@st.cache_data(ttl=15)
def api_alive() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.ok and r.json().get("model_loaded", False)
    except requests.RequestException:
        return False


def options(bundle: dict) -> dict[str, list[str]]:
    """Category levels the model saw at least 30 times (anything else is treated as 'infrequent')."""
    enc = bundle["preprocessor"].named_transformers_["cat"]
    out = {}
    for col, cats, infreq in zip(bundle["categorical"], enc.categories_, enc.infrequent_categories_):
        rare = set(infreq) if infreq is not None else set()
        out[col] = [c for c in cats if c not in rare and c != "Unknown"]
    return out


def score(records: list[dict]) -> tuple[list[dict], str]:
    """Score through the API when it is up, otherwise in-process. Returns (predictions, backend label)."""
    payload = [{**r, "published_at": str(r["published_at"]), "close_at": str(r["close_at"])} for r in records]
    if api_alive():
        r = requests.post(f"{API_URL}/predict/batch", json={"tenders": payload}, timeout=60)
        if r.ok:
            return r.json()["predictions"], f"FastAPI service at {API_URL}"
        st.warning(f"API returned {r.status_code}: {r.text[:200]}. Scoring in-process instead.")
    df = pd.DataFrame(payload)
    df["closing_extended"] = df["closing_extended"].astype(int)
    return score_tenders(df, local_bundle()), "in-process (committed bundle)"


# --------------------------------------------------------------------------------------------- widgets
def gauge(risk: float, band: str, cutoffs: dict) -> go.Figure:
    hi, med = cutoffs["High"] * 100, cutoffs["Medium"] * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100, number={"suffix": "%", "font": {"size": 44, "color": BAND_COLOR[band]}},
        title={"text": "Risk of not reaching award", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickvals": [0, round(med), round(hi), 50, 100]},
            "bar": {"color": BAND_COLOR[band], "thickness": 0.28},
            "steps": [{"range": [0, med], "color": "rgba(46,139,87,0.18)"},
                      {"range": [med, hi], "color": "rgba(224,138,30,0.20)"},
                      {"range": [hi, 100], "color": "rgba(192,57,43,0.18)"}],
        },
    ))
    fig.update_layout(height=260, margin={"l": 30, "r": 30, "t": 60, "b": 10})
    return fig


def reasons_chart(feats: list[dict]) -> go.Figure:
    labels = [FEATURE_LABEL.get(f["feature"], f["feature"]) for f in feats][::-1]
    vals = [f["contribution"] for f in feats][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=[GREEN if v > 0 else RED for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition="inside", insidetextanchor="middle",
        textfont={"color": "white", "size": 13},
    ))
    lim = max(abs(v) for v in vals) * 1.15 or 1
    fig.add_vline(x=0, line_color=GREY)
    fig.update_layout(height=70 + 48 * len(feats), margin={"l": 10, "r": 20, "t": 10, "b": 30},
                      xaxis={"title": "SHAP contribution to the log-odds of award", "range": [-lim, lim]},
                      yaxis={"automargin": True})
    return fig


def band_pill(band: str) -> str:
    return f'<span class="band-pill" style="background:{BAND_COLOR[band]}">{band} risk</span>'


def show_prediction(pred: dict, cutoffs: dict, threshold: float):
    band = pred["risk_band"]
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(gauge(pred["failure_risk"], band, cutoffs), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown(band_pill(band), unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Award probability", f"{pred['award_probability']:.0%}")
        m2.metric("Flagged", "Yes" if pred["flagged_at_risk"] else "No")
        st.caption(f"Bands come from the training-set distribution of risk: Medium above "
                   f"{cutoffs['Medium']:.0%}, High above {cutoffs['High']:.0%}. The flag applies the "
                   f"macro-F1 threshold chosen on the 2020 validation year (award probability below {threshold:.2f}).")
    st.markdown("**Why this score** (the three features that moved it most)")
    for f in pred["leading_features"]:
        colour = GREEN if f["contribution"] > 0 else RED
        verb = "raises" if f["contribution"] > 0 else "lowers"
        st.markdown(f'<div class="reason" style="border-left-color:{colour}"><b>'
                    f'{FEATURE_LABEL.get(f["feature"], f["feature"])}</b> {verb} the chance of award '
                    f'({f["contribution"]:+.2f} log-odds)</div>', unsafe_allow_html=True)
    st.plotly_chart(reasons_chart(pred["leading_features"]), use_container_width=True,
                    config={"displayModeBar": False})


# --------------------------------------------------------------------------------------------- app
bundle = local_bundle()
OPTS = options(bundle)
CUTOFFS, THRESHOLD = bundle["risk_band_cutoffs"], bundle["threshold"]

with st.sidebar:
    st.markdown(f"### <span style='color:{NAVY}'>Tender Award Risk</span>", unsafe_allow_html=True)
    st.caption("Early warning for tenders on Kenya's Public Procurement Information Portal")
    if api_alive():
        st.success(f"API online · {API_URL}", icon="✅")
    else:
        st.warning("API offline · scoring in-process", icon="⚠️")
    st.markdown("**Deployed model**")
    st.markdown(f"- {bundle['model_name']}\n- outcome: {bundle['outcome_definition']}\n"
                f"- trained on {bundle['n_train']:,} tenders ({bundle['trained_on']})\n"
                f"- scikit-learn {bundle['versions']['scikit-learn']}")
    st.markdown("[Repository](https://github.com/Emmanuelbett67/ppip-tender-award-risk) · "
                "[Report](https://github.com/Emmanuelbett67/ppip-tender-award-risk/blob/main/reports/REPORT.pdf)")
    st.caption("DSA 8401 capstone · Emmanuel Bett · Strathmore University")

st.title("Will this tender reach award?")
st.markdown("Score a tender **before it is advertised**, using only what is known at publication. "
            "Bidder counts, evaluation results and contract data are never used: they do not exist yet.")

tab_score, tab_batch, tab_evidence = st.tabs(["Score a tender", "Batch scoring", "Model evidence"])

# ---------------------------------------------------------------- tab 1: single tender + what-if
with tab_score:
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        with st.form("tender"):
            st.markdown("#### Draft tender")
            title = st.text_input("Title", "Supply and delivery of laboratory reagents and consumables")
            description = st.text_area("Description (optional)", "Framework contract for the supply of reagents "
                                       "to the county referral hospital.", height=70)
            a, b = st.columns(2)
            method = a.selectbox("Procurement method", OPTS["procurement_method"],
                                 index=OPTS["procurement_method"].index("Open Tender"))
            category = b.selectbox("Category", OPTS["procurement_category"],
                                   index=OPTS["procurement_category"].index("Goods"))
            pe_type = a.selectbox("Procuring-entity type", OPTS["pe_type"],
                                  index=OPTS["pe_type"].index("County Government"))
            county = b.selectbox("County", OPTS["county"], index=OPTS["county"].index("Nakuru"))
            published = a.date_input("Publication date", date(2024, 3, 4))
            close = b.date_input("Closing date", date(2024, 3, 25))
            validity = a.number_input("Validity period (days)", 0, 2000, 120)
            fee = b.number_input("Tender fee (KES, 0 if free)", 0, 1_000_000, 1000, step=500)
            addenda = a.number_input("Addenda issued so far", 0, 50, 0)
            prior = b.number_input("Entity's prior tenders on PPIP", 0, 100_000, 350)
            extended = st.toggle("Closing date has been extended", False)
            submitted = st.form_submit_button("Score this tender", type="primary", use_container_width=True)

    record = {
        "title": title, "description": description or None, "procurement_method": method,
        "procurement_category": category, "pe_type": pe_type, "county": county,
        "published_at": published, "close_at": close, "validity_in_days": int(validity), "tender_fee": float(fee),
        "n_addenda_pre_close": int(addenda), "closing_extended": bool(extended), "pe_prior_tenders": int(prior),
    }

    with right:
        if close < published:
            st.error("The closing date is before the publication date.")
        elif submitted or "last" in st.session_state:
            if submitted:
                # what-if scenarios scored in the same call as the tender itself
                variants = {"As drafted": record}
                if (close - published).days < 21:
                    variants["Bidding window of 21 days"] = {**record, "close_at": published + timedelta(days=21)}
                variants["Bidding window + 14 days"] = {**record, "close_at": close + timedelta(days=14)}
                if method != "Open Tender":
                    variants["Open Tender instead"] = {**record, "procurement_method": "Open Tender"}
                if fee > 0:
                    variants["No tender fee"] = {**record, "tender_fee": 0.0}
                else:
                    variants["Tender fee of KES 1,000"] = {**record, "tender_fee": 1000.0}
                if validity < 120:
                    variants["Validity of 120 days"] = {**record, "validity_in_days": 120}
                with st.spinner("Scoring…"):
                    preds, backend = score(list(variants.values()))
                st.session_state["last"] = {"names": list(variants), "preds": preds, "backend": backend}
            last = st.session_state["last"]
            base = last["preds"][0]
            show_prediction(base, CUTOFFS, THRESHOLD)
            st.caption(f"Scored by the {last['backend']}.")

            st.markdown("#### What would change the score?")
            st.caption("The same tender with one thing changed, scored the same way. Positive change means a "
                       "higher chance of award.")
            rows = []
            for name, p in zip(last["names"][1:], last["preds"][1:]):
                rows.append({"Change": name, "Award probability": p["award_probability"],
                             "Change (pp)": round((p["award_probability"] - base["award_probability"]) * 100, 1),
                             "Risk band": p["risk_band"]})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
                    "Award probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                    "Change (pp)": st.column_config.NumberColumn(format="%+.1f pp"),
                })
            with st.expander("Request sent to the API"):
                st.code(json.dumps({**record, "published_at": str(published), "close_at": str(close)}, indent=2),
                        language="json")
        else:
            st.info("Fill in the draft tender on the left and press **Score this tender**.", icon="👈")

# ---------------------------------------------------------------- tab 2: batch
with tab_batch:
    st.markdown("Upload a CSV with one tender per row (same columns as the API request), or use the example file.")
    example_path = ROOT / "data" / "examples" / "draft_tenders.csv"
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader("CSV of draft tenders", type="csv", label_visibility="collapsed")
    use_example = c2.button("Use the example file", use_container_width=True)
    src = None
    if upload is not None:
        src = pd.read_csv(upload)
    elif use_example or st.session_state.get("batch_example"):
        st.session_state["batch_example"] = True
        src = pd.read_csv(example_path)
    if src is not None:
        src = src.copy()
        if "closing_extended" in src:
            src["closing_extended"] = src["closing_extended"].astype(bool)
        src = src.where(pd.notna(src), None)
        recs = src.to_dict("records")
        with st.spinner(f"Scoring {len(recs)} tenders…"):
            preds, backend = score(recs)
        out = src.copy()
        out["award_probability"] = [p["award_probability"] for p in preds]
        out["failure_risk"] = [p["failure_risk"] for p in preds]
        out["risk_band"] = [p["risk_band"] for p in preds]
        out["flagged_at_risk"] = [p["flagged_at_risk"] for p in preds]
        out["top_reason"] = [FEATURE_LABEL.get(p["leading_features"][0]["feature"], p["leading_features"][0]["feature"])
                             if p["leading_features"] else "" for p in preds]
        counts = out["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tenders scored", len(out))
        k2.metric("High risk", int(counts["High"]))
        k3.metric("Medium risk", int(counts["Medium"]))
        k4.metric("Low risk", int(counts["Low"]))
        show_cols = ["title", "procurement_method", "pe_type", "county", "award_probability", "risk_band", "top_reason"]
        st.dataframe(
            out.sort_values("award_probability")[[c for c in show_cols if c in out]],
            hide_index=True, use_container_width=True,
            column_config={"award_probability": st.column_config.ProgressColumn(
                "Award probability", format="percent", min_value=0, max_value=1)},
        )
        st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_tenders.csv", "text/csv")
        st.caption(f"Scored by the {backend}.")

# ---------------------------------------------------------------- tab 3: evidence
with tab_evidence:
    st.markdown("Forward-chaining temporal validation: train on 2018 to Y−1, test on Y, for Y = 2021 to 2024. "
                "Primary outcome definition: awarded versus terminated-or-stalled (108,527 tenders).")
    pooled = pd.read_csv(REPORT_DIR / "metrics" / "metrics_pooled.csv")
    ci = pd.read_csv(REPORT_DIR / "metrics" / "bootstrap_ci.csv") if (REPORT_DIR / "metrics" / "bootstrap_ci.csv").exists() else None
    d2 = pooled[pooled["definition"] == "D2"][["model", "ROC-AUC", "F1 (non-award)", "Balanced acc."]]
    d2 = d2.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    st.dataframe(d2, hide_index=True, use_container_width=True, column_config={
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "F1 (non-award)": st.column_config.NumberColumn(format="%.3f"),
        "Balanced acc.": st.column_config.NumberColumn(format="%.3f"),
    })
    figs = [("06_auc_by_year.png", "AUC by test year: the tree ensembles lead every year."),
            ("07_auc_by_definition.png", "AUC depends on how failure is labelled: 0.73 naive, 0.63 observed "
                                         "activity, 0.84 documented terminations."),
            ("12_shap_importance_deployed.png", "What drives the deployed model (mean |SHAP| on 2024 tenders)."),
            ("10_fairness_agpo.png", "Fairness audit: AGPO-reserved tenders are flagged less often and calibrated equally.")]
    g1, g2 = st.columns(2)
    for i, (name, cap) in enumerate(figs):
        path = REPORT_DIR / "figures" / name
        if path.exists():
            (g1 if i % 2 == 0 else g2).image(str(path), caption=cap, use_container_width=True)
    with st.expander("Deployed model metadata"):
        meta_path = MODEL_PATH.parent / "metadata.json"
        if meta_path.exists():
            st.json(json.loads(meta_path.read_text()))
