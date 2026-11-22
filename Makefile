PY ?= python

.PHONY: install data sample-data evaluate evaluate-fast train test serve ui docker-build docker-up

install:
	$(PY) -m pip install -e ".[dev]"

## full pipeline (needs the full extracts in data/raw)
data:
	$(PY) scripts/make_dataset.py --raw data/raw --out data/processed/tenders_features.parquet

## same pipeline on the committed 5% sample (runs anywhere, ~1 min)
sample-data:
	$(PY) scripts/make_dataset.py --raw data/sample --out data/processed/sample_features.parquet

evaluate:
	$(PY) scripts/evaluate.py

evaluate-fast:
	$(PY) scripts/evaluate.py --fast

train:
	$(PY) scripts/train.py

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(PY) -m streamlit run app/streamlit_app.py

docker-build:
	docker build -t ppip-tender-award-risk .

docker-up:
	docker compose up --build
