# F1 Race Intelligence Platform — API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

**Authentication**: The local development API does not require an API key. For production deployment, ensure CORS and rate limiting are configured.

---

## Health

### `GET /health`
Health check for monitoring.

**Response:**
```json
{"status": "healthy", "version": "1.0.0"}
```

---

## Session Endpoints

### `GET /api/sessions`
List all processed sessions.

**Response:**
```json
{
  "sessions": {
    "2024_Italy_R": {
      "year": 2024, "gp": "Italy", "session_type": "R",
      "total_frames": 32000, "frame_rate": 4.0, "drivers": 20
    }
  }
}
```

---

### `GET /api/session`
Fetch / process a session. Returns metadata + track outline.

**Query Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `year` | int | ✓ | Season year (2018–2030) |
| `gp` | string | ✓ | Grand Prix name (e.g., "Italy", "Monaco") |
| `session` | string | | Session type: R, Q, FP1, FP2, FP3, S, SQ (default: R) |

**Response:** `SessionMetadata` with embedded `track` array (< 200KB).

**Notes:**
- First call triggers the full pipeline (fetch → resample → validate → cache). May take 30-60s.
- Subsequent calls return cached data instantly.

---

### `GET /api/session/{session_id}/telemetry`
Get a chunk of synchronized telemetry frames (gzipped).

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `start` | float | 0.0 | Start time in seconds |
| `duration` | float | 30.0 | Chunk duration (1–120s) |

**Response:** Gzipped JSON with compact keys:
```json
{
  "session_id": "2024_Italy_R",
  "start": 120.0,
  "duration": 30.0,
  "frame_rate": 4.0,
  "frame_count": 120,
  "frames": [
    {
      "t": 120.0,
      "d": {
        "VER": {"x": 123.4, "y": 567.8, "d": 3520.4, "s": 298.2, "th": 87.0, "br": 0.0, "g": 8, "drs": true, "tc": "M", "ta": 8, "l": 12, "syn": false},
        "HAM": { ... }
      }
    }
  ]
}
```

**Compact key mapping:**
| Key | Full Name |
|-----|-----------|
| `d` | distance (m) |
| `s` | speed (km/h) |
| `th` | throttle (0-100) |
| `br` | brake (0-1) |
| `g` | gear |
| `tc` | tyre compound |
| `ta` | tyre age (laps) |
| `l` | lap number |
| `syn` | synthetic flag |

---

### `GET /api/session/{session_id}/events`
Get all detected events with ML scores.

**Response:**
```json
{
  "session_id": "2024_Italy_R",
  "events": [
    {
      "t": 452.3, "type": "overtake", "driver": "VER",
      "details": {"victim": "HAM", "proximity_m": 12.5},
      "highlightScore": 0.85, "confidence": 0.92, "source": "rule"
    }
  ]
}
```

**Event types:** `overtake`, `pit_stop`, `fastest_lap`, `incident`

---

## Prediction Engine

### `POST /api/predict`
Run a natural-language prediction query against the Tier-2 race simulator.

**Request body:**
```json
{
  "session_id": "2024_Monaco_Grand_Prix_R",
  "query": "who will win the race?",
  "strategy_overrides": {}
}
```

**Response:**
```json
{
  "scenario_type": "race_projection",
  "snapshot": {
    "ml_deg_model_active": true,
    "track_temp_c": 38.5,
    "top10_drivers": [...]
  },
  "scenario": {
    "final_order": [...],
    "confidence": 0.72,
    "laps_simulated": 38
  }
}
```

**Supported intents:**
| Query phrase | Scenario type |
|---|---|
| "who wins", "project the race" | `race_projection` |
| "1-stop vs 2-stop", "compare strategy" | `strategy_comparison` |
| "safety car now", "VSC deployed" | `safety_car` |
| "should he pit", "bring him in" | `pit_stop` |
| "tyre deg", "how long can he survive" | `tyre_degradation` |
| "can X overtake Y", "gap closing" | `overtake_window` |

> `ml_deg_model_active: true` indicates predictions are XGBoost ML-backed (temperature and driver adjusted). `false` indicates static compound table fallback.

---

## Chat

### `POST /api/chat`
Send a message to the RAG-powered AI Pit Crew.

**Request body:**
```json
{"session_id": "2024_Monaco_Grand_Prix_R", "message": "What was Leclerc's best lap?"}
```

**Response:**
```json
{"reply": "Charles Leclerc set his best lap of 1:13.166 on lap 52, running on a fresh Hard compound."}
```

---

## WebSocket (Future)

### `WS /ws/session/{session_id}`
Push live telemetry updates.

**Message format:**
```json
{"type": "telemetry_chunk", "start": 450.0, "duration": 5, "frames": [...]}
{"type": "event", "event_type": "overtake", "t": 452.3, "driver": "VER"}
```
