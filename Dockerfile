FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# dependencies first so the layer is cached between code changes
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[ui]"

# API code, Streamlit front-end, the serialized model bundle and the evidence the UI shows
COPY api ./api
COPY app ./app
COPY models/award_risk_v2 ./models/award_risk_v2
COPY data/examples ./data/examples
COPY reports/metrics ./reports/metrics
COPY reports/figures ./reports/figures

ENV MODEL_PATH=/app/models/award_risk_v2/award_risk_bundle.joblib
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
