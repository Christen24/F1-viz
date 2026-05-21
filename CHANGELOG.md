# Changelog

All notable changes to this project are documented here.

---

## [Unreleased]

### Added
- WebSocket live telemetry push endpoint (planned)
- Multi-session comparison view (planned)

---

## [1.3.1] — 2026-05-19

### Fixed
- `Dockerfile.backend`: `apt-get install` now retries up to 3 times with a 5-second delay to handle transient `deb.debian.org` network failures during Docker builds.
- `docker-compose up -d --build` no longer fails on intermittent package registry timeouts.

---

## [1.3.0] — 2026-05-13

### Added
- **XGBoost Tyre Degradation Model** (`src/backend/services/tyre_deg_model.py`)
  - Trained on 45,110 laps across 60 races (2022–2024)
  - Predicts compound-specific deg rates adjusted for track temperature, air temperature, driver style, circuit, and fuel load
  - Lazy-loaded — falls back to static table silently when model file is absent
- **ML Training Script** (`scripts/ml/train_tyre_deg.py`)
  - Downloads and caches FastF1 race data automatically
  - Outputs `tyre_deg_xgb.joblib`, `tyre_deg_meta.joblib`, `tyre_deg_report.json` to `data/models/`
- `track_temp_c` field added to `RaceSnapshot` dataclass
  - Resolved from `live_context` → `metadata` → lap weather → default 30.0°C
- `ml_deg_model_active` boolean added to snapshot API response
- 45 new tests in `tests/test_tyre_deg_ml.py` (all passing)

### Changed
- `docker-compose.yml`: data volume changed from named Docker volume to local bind mount (`./data:/app/data`) so host-trained ML models are visible inside the container

---

## [1.2.0] — 2026-05-09

### Added
- **Fuel-Corrected Degradation Regression**: Subtracts 0.035 s/lap fuel-burn gain from raw lap times before running the deg slope regression. Eliminates artificial inflation of early-stint deg rates.
- **Dirty Air Proximity Penalty**: Linear pace penalty (up to 0.35 s/lap) applied to cars following within 0.8 s of the car ahead. Fixes clean-air ghosting in close-gap simulations.
- **Cold Tyre Out-Lap Penalty**: 1.5 s penalty on the lap immediately after a pit stop. Models the real-world grip ramp-up phase.
- 29 new tests in `tests/test_prediction_engine_physics.py` (all passing)

### Changed
- Simulation accuracy rating improved from **5.8 → 7.1**
- `test_forced_override_changes_result` updated to test VER override (not LEC, who is typically the leader)

---

## [1.1.0] — 2026-04-21

### Added
- **Tier-2 Prediction Engine** (`src/backend/services/prediction_engine.py`)
  - `simulate_race_to_finish()`: Lap-by-lap projection of all drivers to the chequered flag
  - `simulate_strategy_comparison()`: Head-to-head delta table for multiple pit strategies
  - 6 intent types: `race_projection`, `strategy_comparison`, `safety_car`, `pit_stop`, `tyre_degradation`, `overtake_window`
- 86 tests in `tests/test_prediction_engine_tier2.py` (all passing)

---

## [1.0.0] — 2026-04-07

### Added
- FastF1 telemetry ingestion and caching pipeline
- 3D track visualization with real-time car positioning
- RAG-powered AI Pit Crew chatbot (pgvector + Ollama/Gemini)
- YouTube video synchronization with telemetry playback
- Docker Compose deployment (backend + frontend + postgres)
