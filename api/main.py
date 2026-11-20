"""FastAPI service: a procuring entity submits a draft tender and receives an award probability, a risk band
and the three features that most moved the score.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from tender_risk import __version__
from tender_risk.config import BUNDLE_PATH
from tender_risk.scoring import load_bundle, score_tenders

MODEL_PATH = Path(os.environ.get("MODEL_PATH", BUNDLE_PATH))
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bundle"] = load_bundle(MODEL_PATH)
    yield
    STATE.clear()


app = FastAPI(
    title="Tender Award Risk API",
    version=__version__,
    description="Early-warning score for tenders on Kenya's Public Procurement Information Portal (PPIP). "
                "Send the information available when a tender is published; receive the probability that it "
                "reaches contract award, a Low/Medium/High risk band and the leading contributing features.",
    lifespan=lifespan,
)


class DraftTender(BaseModel):
    """Information known at publication. Nothing that only exists after the closing date is accepted."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "title": "Supply and delivery of laboratory reagents and consumables",
        "description": "Framework contract for the supply of reagents to the county referral hospital.",
        "procurement_method": "Open Tender",
        "procurement_category": "Goods",
        "pe_type": "County Government",
        "county": "Nakuru",
        "published_at": "2024-03-04",
        "close_at": "2024-03-25",
        "validity_in_days": 120,
        "tender_fee": 1000,
        "n_addenda_pre_close": 0,
        "closing_extended": False,
        "pe_prior_tenders": 350,
    }})

    title: str = Field(..., min_length=1, max_length=500, description="Tender title as advertised")
    description: Optional[str] = Field(None, max_length=5000, description="Tender description (only its presence is used)")
    procurement_method: str = Field(..., description="e.g. Open Tender, Request for Quotations, Restricted Tendering")
    procurement_category: str = Field(..., description="Goods, Works, Non Consultancy Services or Consultancy Services")
    pe_type: str = Field(..., description="Procuring-entity type, e.g. County Government, State Corporation")
    county: str = Field(..., description="County of the procuring entity")
    published_at: date
    close_at: date
    validity_in_days: Optional[int] = Field(None, ge=0, le=2000, description="Tender validity period in days")
    tender_fee: Optional[float] = Field(None, ge=0, description="Tender document fee in KES (0 or null if free)")
    n_addenda_pre_close: int = Field(0, ge=0, description="Addenda issued so far")
    closing_extended: bool = Field(False, description="Closing date has been extended by an addendum")
    pe_prior_tenders: Optional[int] = Field(None, ge=0, description="Tenders the entity has published before this one")


class LeadingFeature(BaseModel):
    feature: str
    contribution: float = Field(..., description="SHAP contribution in log-odds of award (+ raises, - lowers)")
    direction: str


class Prediction(BaseModel):
    award_probability: float = Field(..., description="P(tender reaches contract award | observed activity)")
    failure_risk: float = Field(..., description="1 - award_probability")
    risk_band: str = Field(..., description="Low / Medium / High, from training-set risk quantiles (50th, 80th)")
    flagged_at_risk: bool = Field(..., description="award_probability below the model's classification threshold")
    leading_features: list[LeadingFeature]


class BatchRequest(BaseModel):
    tenders: list[DraftTender] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    n: int
    predictions: list[Prediction]
    band_counts: dict[str, int]


def _frame(tenders: list[DraftTender]) -> pd.DataFrame:
    df = pd.DataFrame([t.model_dump() for t in tenders])
    df["closing_extended"] = df["closing_extended"].astype(int)
    return df


def _score(tenders: list[DraftTender]) -> list[dict]:
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        return score_tenders(_frame(tenders), bundle)
    except Exception as exc:  # scoring errors are client-visible but never leak a stack trace
        raise HTTPException(status_code=422, detail=f"could not score tender(s): {exc}") from exc


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok", "model_loaded": "bundle" in STATE, "version": __version__}


@app.get("/model", tags=["service"])
def model_info():
    b = STATE.get("bundle")
    if b is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {k: v for k, v in b.items() if k not in ("preprocessor", "model", "_explainer")}


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(tender: DraftTender):
    return _score([tender])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(req: BatchRequest):
    preds = _score(req.tenders)
    counts = {b: sum(p["risk_band"] == b for p in preds) for b in ("Low", "Medium", "High")}
    return {"n": len(preds), "predictions": preds, "band_counts": counts}
