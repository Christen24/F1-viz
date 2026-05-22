# Contributing to F1 Race Intelligence Platform

Thank you for your interest in contributing. This guide covers local setup, the testing workflow, and commit conventions.

---

## Local Setup

### Prerequisites
- Python 3.11
- Docker Desktop
- Node.js 20+
- Git

### Backend
```bash
pip install -r requirements.txt
uvicorn src.backend.app:app --reload
```

### Frontend
```bash
cd src/frontend
npm install
npm run dev
```

### Full Stack (Docker)
```bash
docker-compose up -d --build
```

---

## Running Tests

Run the full test suite:
```bash
pytest -v
```

Run specific modules:
```bash
# Tier-2 race simulator
pytest tests/test_prediction_engine_tier2.py -v

# Physics engine (fuel, dirty air, cold tyre)
pytest tests/test_prediction_engine_physics.py -v

# ML tyre degradation model
pytest tests/test_tyre_deg_ml.py -v
```

All tests must pass before opening a pull request.

---

## ML Model Training

To retrain the XGBoost tyre degradation model from fresh FastF1 data:
```bash
python scripts/ml/train_tyre_deg.py
```

Expected targets after training:
- MAE < 0.008 s/lap
- Improvement over baseline > 60%
- `track_temp` in top 3 feature importances

Check results:
```bash
python -c "import json; print(json.dumps(json.load(open('data/models/tyre_deg_report.json')), indent=2))"
```

---

## Commit Conventions

Use the following prefixes to keep the history readable:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `physics:` | Simulation constants or model logic |
| `ml:` | XGBoost model, training script, or TyreDegModel |
| `test:` | New or updated tests |
| `docs:` | README, API.md, CHANGELOG, or docstrings |
| `deps:` | requirements.txt changes |
| `docker:` | Dockerfile or docker-compose changes |
| `config:` | Environment, settings, or CI changes |
| `refactor:` | Code restructuring with no behavior change |

Example:
```
ml: add temperature sensitivity validation to TyreDegModel
```

---

## Code Style

- **Python**: Follow PEP 8 guidelines. Type hints are highly encouraged for all backend services and API routes.
- **TypeScript**: Use strict typing. Avoid `any` where possible.
- **Formatting**: We use `ruff` for Python and `prettier` for Frontend formatting.

---

## Pull Request Checklist

- [ ] Tests pass (`pytest -v`)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Relevant docstrings updated
- [ ] No hardcoded API keys or credentials
