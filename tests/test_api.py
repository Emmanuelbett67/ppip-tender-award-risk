import pytest
from fastapi.testclient import TestClient

from tender_risk.config import BUNDLE_PATH

pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model bundle not trained yet")

EXAMPLE = {
    "title": "Supply and delivery of laboratory reagents and consumables",
    "description": "Framework contract for reagents.",
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
}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info_has_no_estimators(client):
    body = client.get("/model").json()
    assert "threshold" in body and "model" not in body and "preprocessor" not in body


def test_predict_single(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["award_probability"] <= 1
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert len(body["leading_features"]) == 3


def test_predict_rejects_bad_payload(client):
    bad = {**EXAMPLE, "published_at": "not a date"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"tenders": [EXAMPLE, {**EXAMPLE, "procurement_method": "Direct Procurement"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2 and sum(body["band_counts"].values()) == 2
