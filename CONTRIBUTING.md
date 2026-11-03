# Contributing to DefectLens AI

Thank you for your interest in contributing to DefectLens AI — an industrial visual inspection system for detecting manufacturing defects.

## Table of Contents
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Test Requirements](#test-requirements)
- [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

1. **Fork the repository** on GitHub.

2. **Clone your fork:**
   ```bash
   git clone https://github.com/<your-username>/defectlens-ai.git
   cd defectlens-ai
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   # or
   make install-dev
   ```

5. **Verify the test suite passes before making changes:**
   ```bash
   make test
   ```

---

## Project Structure

```
defectlens-ai/
├── src/defectlens/
│   ├── data/           # Dataset loading, transforms, validation
│   ├── models/         # Classifier, anomaly detector, segmentor
│   ├── training/       # Training loop and callbacks
│   ├── evaluation/     # Metrics and error analysis
│   ├── inference/      # Predictor and benchmarking
│   └── visualization/  # Result visualizers
├── app/
│   ├── streamlit_app.py  # Interactive Streamlit dashboard
│   └── api.py            # FastAPI REST endpoints
├── scripts/            # Training, evaluation, inference scripts
├── configs/            # YAML configuration files
├── tests/              # Unit tests (pytest)
├── notebooks/          # Jupyter EDA and training notebooks
└── docs/               # Architecture and API documentation
```

---

## Code Standards

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Maximum line length: **100 characters**.
- **Formatting:** Use `black` (`make format`) before committing.
- **Imports:** Sort with `isort` (`make format`).
- **Docstrings:** All public classes and functions **must** have Google-style docstrings.
- **Type hints:** All new functions should include type annotations.
- **No fabricated results:** Do not add hardcoded evaluation numbers or fake model outputs.

---

## Test Requirements

- All new features **must** include corresponding unit tests.
- All tests must pass: `make test`
- Do not lower existing test coverage.
- Use `pytest.mark.skip` (with a reason) for tests that require external datasets or GPU.
- FastAPI tests use `pytest.importorskip("fastapi")` — install fastapi to run them.

---

## Submitting Changes

1. Ensure all tests pass: `make test`
2. Run lint checks: `make lint`
3. Update `docs/` if your change affects architecture or API behaviour.
4. Push your branch and open a **Pull Request** with:
   - A clear description of the change and motivation.
   - Screenshots or example output where applicable.
   - Reference to any related issue (`Fixes #123`).

---

## Reporting Issues

Open a GitHub Issue with:
- A minimal reproducible example.
- Your Python version and OS.
- The full error traceback.

We appreciate all contributions — bug reports, documentation improvements, and new features alike.
