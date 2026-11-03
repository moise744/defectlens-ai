# DefectLens AI — Quick Start Guide

Get up and running in under 5 minutes.

## Prerequisites

- Python 3.9+
- pip

## 1. Clone and Install

```bash
git clone https://github.com/moise744/defectlens-ai.git
cd defectlens-ai
pip install -e ".[dev]"
```

## 2. Verify Installation

```bash
python -m pytest tests/ -q
# Expected: 36 passed
```

## 3. Run the Streamlit Dashboard (no dataset needed)

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

Upload any industrial surface image (JPEG/PNG) and select an inspection mode.

## 4. Run the FastAPI Server

```bash
pip install fastapi uvicorn[standard]
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# API docs at http://localhost:8000/docs
```

## 5. Run Inference on a Single Image

```bash
python scripts/inference.py --image path/to/image.jpg --mode classifier
```

## 6. Download the Dataset and Train

```bash
# Download MVTec AD (see data/README.md for instructions)
python scripts/download_dataset.py --category bottle

# Train the classifier
python scripts/train.py --config configs/train_classifier.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/best_model.pth
```

## 7. Docker

```bash
docker build -t defectlens-ai .
docker run -p 8501:8501 -p 8000:8000 defectlens-ai
```

## Key Commands Reference

| Command | Purpose |
|---------|---------|
| `make install` | Install production deps |
| `make install-dev` | Install dev deps |
| `make test` | Run all tests |
| `make run-app` | Launch Streamlit dashboard |
| `make run-api` | Launch FastAPI server |
| `make lint` | Run linter |
| `make docker-build` | Build Docker image |
