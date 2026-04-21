# Fake News ML Pipeline

Production-style tabular ML pipeline for fake-news classification with:

- Data profiling and anomaly checks
- Training pipeline with sentinel fix (`caps_ratio == 0 -> NaN`) inside sklearn pipeline
- FastAPI inference service (`/predict`, `/predict_batch`, `/health`)

## Project Structure

- `fake_news_pipeline/`
- `fake_news_pipeline/profiling.py` data audit checks
- `fake_news_pipeline/modeling.py` training, evaluation, model export
- `fake_news_pipeline/api.py` FastAPI service and model loading
- `Script_1.py` compatibility profiling entrypoint
- `Script_2.py` compatibility training entrypoint
- `Script_3.py` compatibility API entrypoint
- `3_serve.py` uvicorn entrypoint module

## Setup

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Expected training dataset path:

- `32_fake_news.csv` in project root

Required columns:

- `id`
- `year`
- `word_count`
- `caps_ratio`
- `exclamation_count`
- `source_credibility_rank`
- `fake`

## Run Profiling

```bash
python Script_1.py
```

## Train and Evaluate

```bash
python Script_2.py
```

Optional custom paths:

```bash
python Script_2.py --data-path path/to/data.csv --model-path path/to/model.joblib --plots-dir path/to/plots
```

## Serve API

```bash
uvicorn 3_serve:app --host 0.0.0.0 --port 8000
```

## Example Requests

Single:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"id":1,"year":2023,"word_count":512,"caps_ratio":0.07,"exclamation_count":4,"source_credibility_rank":22}'
```

Batch:

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"articles":[{"id":1,"year":2023,"word_count":512,"caps_ratio":0.07,"exclamation_count":4,"source_credibility_rank":22}]}'
```
