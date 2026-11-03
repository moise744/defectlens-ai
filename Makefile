.PHONY: install install-dev test lint format run-app run-api docker-build docker-run clean help

PROJECT_NAME = defectlens-ai
PORT_APP     = 8501
PORT_API     = 8000

help:
	@echo "DefectLens AI — available targets:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install dev dependencies (includes test tools)"
	@echo "  test          Run the full test suite"
	@echo "  lint          Run flake8 linter"
	@echo "  format        Auto-format code with black + isort"
	@echo "  run-app       Launch Streamlit inspection dashboard"
	@echo "  run-api       Launch FastAPI REST server"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container (app + api)"
	@echo "  clean         Remove caches and build artefacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

lint:
	flake8 src/ app/ scripts/ --max-line-length=100 --ignore=E203,W503

format:
	black src/ app/ scripts/ tests/ --line-length 100
	isort src/ app/ scripts/ tests/

run-app:
	streamlit run app/streamlit_app.py --server.port $(PORT_APP)

run-api:
	uvicorn app.api:app --host 0.0.0.0 --port $(PORT_API) --reload

docker-build:
	docker build -t $(PROJECT_NAME):latest .

docker-run:
	docker run -p $(PORT_APP):$(PORT_APP) -p $(PORT_API):$(PORT_API) $(PROJECT_NAME):latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
