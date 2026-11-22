"""Smoke test for the Streamlit front-end: the page renders and a scored tender shows its reasons."""

from pathlib import Path

import pytest

from tender_risk.config import BUNDLE_PATH

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model bundle not trained")


def test_app_renders_and_scores():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    at.button(key="FormSubmitter:tender-Score this tender").click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Why this score" in text
    assert "What would change the score" in text
