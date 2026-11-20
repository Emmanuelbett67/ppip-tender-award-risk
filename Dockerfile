FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# dependencies first so the layer is cached between code changes
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# API code and the serialized model bundle
COPY api ./api
COPY models/award_risk_v2 ./models/award_risk_v2

ENV MODEL_PATH=/app/models/award_risk_v2/award_risk_bundle.joblib
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
