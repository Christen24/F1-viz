# F1 3D Visualization System — API Reference

## Base URL

**Development:** `http://localhost:8000`

---

## Endpoints

### `GET /health`
Health check for monitoring.

**Response:**
```json
{"status": "healthy", "version": "1.0.0"}
```

---

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

## WebSocket (Future)

### `WS /ws/session/{session_id}`
Push live telemetry updates.

**Message format:**
```json
{"type": "telemetry_chunk", "start": 450.0, "duration": 5, "frames": [...]}
{"type": "event", "event_type": "overtake", "t": 452.3, "driver": "VER"}
```
