# Customer Review Analyzer

A one-day data science project: send English customer reviews to a Python API,
inspect sentiment in a Streamlit dashboard, and evaluate a pretrained Hugging Face
model on real Amazon reviews. Run everything locally with Docker Compose.

## First measured result

On the pinned 500-review balanced sample (seed 42), the initial model achieved
**87.0% accuracy**, compared with **50.0%** for the always-negative baseline.
Positive-class precision was 0.894, recall 0.840, and F1 0.866. There were 65 errors:
25 negative reviews predicted positive and 40 positive reviews predicted negative.
No sampled reviews required truncation. These are sample results, not a production guarantee.

Initial verification: 26 automated API/dashboard tests passed, real API sentiment/validation/
truncation checks passed, and both local HTTP endpoints responded successfully.
Dashboard interactions were checked with Streamlit AppTest; visual browser inspection was not performed.

## Start here

Open PowerShell in this folder with Docker Desktop running:

```powershell
docker compose up --build -d
```

The first build downloads Python dependencies, including CPU-only PyTorch.
The first API startup downloads the model (roughly 270 MB) into a named Docker volume.
Allow several minutes. No Hugging Face account, API key, paid inference service, or GPU is required.

- Dashboard: http://localhost:8501
- Interactive API documentation: http://localhost:8000/docs
- Readiness and model information: http://localhost:8000/health

Try a review in **One review**. Then download the example CSV in **Review collection**,
upload it, analyze it, and export predictions. Example reviews are handwritten demo inputs,
not an accuracy benchmark. Extra upload columns are previewed; the exported file contains
only submitted text, label, score, and truncation status.

Useful commands:

```powershell
docker compose ps
docker compose logs --tail 50 api
docker compose down
```

`down` stops and removes this project's containers and network; the model cache stays.
To apply code changes, run `docker compose up --build -d` again.
If the initial model download exceeds the health-check window, inspect the logs and rerun
`docker compose up -d` after the API finishes starting.

## Understand the request path

```text
Browser -> Streamlit dashboard -> HTTP POST -> FastAPI -> tokenizer -> DistilBERT
Browser <- Streamlit dashboard <- JSON      <- FastAPI <- sentiment and score
```

1. **Python** is the programming language for all application and evaluation code.
2. **Hugging Face** hosts the pretrained model and review dataset. We download the
   model, then perform inference locally. Our API is our own service, not Hugging Face's hosted inference API.
3. **Tokenization** turns words and word pieces into numeric IDs that the model accepts.
4. **PyTorch** executes the model and converts its two output values into scores with softmax.
5. **FastAPI** receives JSON, validates it, calls the model, and returns JSON.
6. **Streamlit** provides the browser interface and calls FastAPI with `requests`.
7. **Dockerfile** describes the image: Python, installed packages, source code, and startup command.
8. **Docker Compose** starts two containers from that image with different commands.
   Inside the dashboard container, `api` is the API's network name; `localhost` would refer
   to the dashboard container itself. On your computer you use `localhost:8000` instead.

The model loads once at API startup, not on every request. A lock serializes inference
and internal batches of eight keep memory use bounded. This is a small local demonstration,
not a high-throughput deployment. Ports are exposed only on your computer's loopback interface.

## Try the API directly

```powershell
Invoke-RestMethod http://localhost:8000/health

$reviewBody = @{ text = "The headphones sound great and arrived quickly." } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body $reviewBody

$reviewsBody = @{ texts = @("Excellent quality!", "It broke after one day.") } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict/batch -Method Post -ContentType 'application/json' -Body $reviewsBody
```

| Endpoint | Response |
| --- | --- |
| `GET /health` | Ready state, model ID/revision, input limits |
| `POST /predict` | `label`, `score`, `truncated` |
| `POST /predict/batch` | Ordered `predictions` list |

Inputs must contain nonblank strings, at most 5,000 characters each and 100 reviews per batch.
Invalid inputs receive HTTP 422 and never reach the model. Scores are the model's softmax output
for the chosen class; a score of 0.99 does **not** establish 99% real-world correctness.
Long reviews are truncated to the first 512 tokens, including special tokens, and flagged.

## Download real data and evaluate

Run these commands from this folder. The tools container uses the same Python/model code
and mounts this folder, so downloaded files and reports appear on Windows too:

```powershell
docker compose --profile tools run --build --rm tools python -m scripts.download_data
docker compose --profile tools run --rm tools python -m scripts.evaluate
```

The download script creates:

- `data/development.csv`: 100 reviews from the dataset's train split, 50 per class.
- `data/evaluation.csv`: 500 reviews from its test split, 250 per class.
- `data/manifest.json`: dataset revision, seed, sampling method, and file hashes.

Sampling uses a seeded streaming shuffle with a 10,000-row buffer. This saves downloading
the full dataset, but is **not** a uniform random sample of all reviews. The fixed
revision, algorithm, seed, and recorded hashes make this specific sample auditable.
Keep evaluation data for assessment; explore and make changes using development data.
Separate splits do not guarantee that the source contains no duplicate or near-duplicate reviews.

The evaluator writes `reports/metrics.json`, `predictions.csv`, `errors.csv`, and
`confusion_matrix.png`. It reports accuracy, per-class precision/recall/F1, truncation count,
inference time excluding loading, and an always-majority-class baseline (50% accuracy on this balanced sample).

Open `notebooks/01_evaluation.ipynb` in VS Code to explore the saved data and results.
It contains explanations and exercises; execute it after running both commands above.
To execute it entirely inside Docker:

```powershell
docker compose --profile tools run --rm tools python -m scripts.run_notebook
```

Dataset and report files are ignored by Git. Download scripts and notebook are tracked.
Do not commit the full dataset or model weights. The data and model sources are linked below;
the Hub cards currently list Apache-2.0, and should be consulted for attribution and usage details.

## Tests

```powershell
docker compose --profile tools run --build --rm tools
```

Contract tests use an injected fake predictor so they run offline and test validation,
response structure, and ordering independently of model quality. Streamlit AppTest checks
dashboard behavior and error handling. The evaluation and manual HTTP examples exercise
the real model separately; test fixture scores are not model evaluation results.

With the API and dashboard running, check the real HTTP flow:

```powershell
docker compose --profile tools run --rm tools python -m scripts.smoke_test
```

## Optional: develop with your installed Python

Use Python 3.12 and a separate environment; don't reuse another portfolio project's environment.
You can skip this entire section if using Docker.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

In a second terminal, from the same folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Choose `.venv\Scripts\python.exe` as your notebook kernel. For local CLI evaluation, replace
the Docker command prefix with `.\.venv\Scripts\python.exe`.
Direct dependency versions and model/dataset revisions are pinned. Transitive dependencies
and the Python base-image tag can still change; this is not a complete environment lock.

## Read the code in this order

| File | What to learn |
| --- | --- |
| `app/settings.py` | Shared limits and pinned model/data identifiers |
| `app/model.py` | Tokenization, inference, batching, scores |
| `app/schemas.py` | Input validation and response contracts |
| `app/api.py` | Routes, startup lifecycle, model reuse |
| `dashboard.py` | Forms, session state, HTTP calls, CSV results |
| `scripts/download_data.py` | Streaming, sampling, provenance |
| `scripts/evaluate.py` | Baselines, metrics, error analysis |
| `Dockerfile` / `compose.yaml` | Images, containers, networking, persistent cache |

Suggested one-day learning sequence: setup (30 min), model and data (60 min), API (60 min),
dashboard (90 min), evaluation (60 min), Docker (60 min), tests and README (60 min).
Take a checkpoint after each stage: explain what it does, change one input, predict the result,
then run it. See `LEARNING.md` for exercises.

## Model, dataset, and limitations

- Model: [DistilBERT fine-tuned on SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
  We use existing weights; we do not train or fine-tune a model. It was trained on movie-review
  sentiment, so measuring transfer to product reviews is part of the project.
- Data: [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity).
  One/two-star reviews map to negative (0), four/five-star to positive (1); three-star reviews
  are excluded. Star ratings are imperfect sentiment labels.
- Only English and positive/negative classes are supported. Neutral, mixed, and sarcastic
  statements can receive confident, misleading predictions. Country and demographic bias
  are documented on the model card. Long-review truncation can hide decisive information.
- A balanced 500-review sample cannot establish performance across all product categories
  or the distribution a real business encounters. There is no neutral class, persistence,
  authentication, public hosting, or custom training in this version.
- Reviews are sent to your local API for inference. Hugging Face is contacted for initial
  model/data downloads; the dashboard does not submit review text to a hosted model.

## Troubleshooting

- **Cannot connect to Docker engine:** start Docker Desktop and wait for the engine.
- **Dashboard cannot analyze:** inspect `docker compose ps` and `docker compose logs api`.
- **Model download fails:** check internet access, then `docker compose restart api`.
- **Port occupied:** stop the service using that port or edit the left side of the mapping
  in `compose.yaml`. If changing the API port, also adjust your local example URLs.
- **No evaluation files:** run the download and evaluation commands above first.
