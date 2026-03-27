# F1 Race Intelligence Platform

A modern, high-performance race analysis dashboard for Formula 1 sessions. This platform provides synchronized multi-driver replay, interactive telemetry visualizations, and automated event detection using a "Bento Box" UI architecture.

![F1 Viz](https://img.shields.io/badge/F1-Race%20Intelligence-e10600?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square)
![ECharts](https://img.shields.io/badge/Visualization-ECharts-aa0000?style=flat-square)

## Features

- **2D Session Replay** — Interactive SVG track map with synchronized car tracking and driver labels.
- **Bento Box Dashboard** — Modular "Glassmorphism 2.0" UI for organized telemetry viewing.
- **Gap to Leader Chart** — Dynamic ECharts visualization showing field spreads with real-time playback sync.
- **Integrated Video Stage** — Synchronized YouTube/Local video playback with telemetry-linked timeline.
- **Master Timeline** — Unified 4Hz timeline with cubic spline interpolation for all 20 drivers.
- **Event Detection** — Automatic detection of overtakes, pit stops, and fastest laps.
- **ML Pipeline** — XGBoost-based overtake classifier with automated data labeling.
- **Interactive Toggles** — Quickly toggle individual drivers or teams directly from the visualization panels.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm

### Backend Installation

```bash
cd f1-viz
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Start the FastAPI server
uvicorn src.backend.app:app --reload --port 8000
```

### Frontend Installation

```bash
cd src/frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Project Structure

```
f1-viz/
├── src/
│   ├── backend/       # FastAPI server, data resampling, & exporters
│   │   ├── routers/   # API endpoints (Session, Track, Video)
│   │   ├── services/  # Core logic (FastF1 integration, Lap computation)
│   │   └── data/      # Local telemetry cache and processed datasets
│   ├── frontend/      # React + ECharts dashboard
│   │   ├── src/components/ # UI Components (TrackMap, GapChart, Leaderboard)
│   │   └── src/stores/     # Zustand state management (Playback & Session)
│   └── ml/            # ML training scripts (Dataset prep & Overtake models)
├── tests/             # Backend unit & integration tests
├── scripts/           # Data validation and smoke tests
├── docker-compose.yml
├── .gitignore         # Configured for node_modules, Python venv, and F1 caches
└── README.md
```

## API Reference

The backend provides several key endpoints for session data:
- `GET /api/session/list` — List available historical sessions.
- `GET /api/session/{id}/track-replay` — Fetch SVG track coordinates and synchronized telemetry frames.
- `GET /api/session/{id}/laps` — Fetch lap-by-lap summaries, positions, and events.

## ML Pipeline

The platform includes scripts for training custom event detection models:

```bash
# 1. Prepare dataset from specific session
python src/ml/prepare_dataset.py --session 2024_Bahrain_R

# 2. Train the XGBoost overtake detector
python src/ml/train_overtakes.py --session 2024_Bahrain_R
```

## Legal

This project is intended for research and personal use. All Formula 1 data is the property of Formula One Management (FOM). Data is retrieved via [FastF1](https://github.com/theOehrly/FastF1).
