"""
F1 Race Prediction Engine  —  Tier 1 upgrade
=============================================

What changed from v1:
  + _detect_throttle_cliff()            detects tyre protection mode 2-3 laps
                                         before lap time shows it, using
                                         throttle_pct from laps.json
  + compute_pit_statistics()            replaces static 21s pit loss with
                                         mean ± std from actual pit_duration
                                         events in this session
  + compute_circuit_overtake_score()    replaces uniform probability constants
                                         with a score derived from real drs_pct
                                         and sector time spread for this circuit
  + _max_speed_delta()                  detects lift-and-coast / drag issues
                                         from max_speed trend

  DriverState — 3 new fields:
    throttle_mode       NORMAL | PROTECTING | UNKNOWN
    throttle_drop_pct   how many % points throttle dropped over lookback window
    max_speed_delta     vs 5-lap baseline (negative = slower / lift-and-coast)

  RaceSnapshot — 2 new fields:
    pit_stats           stochastic pit model built from this session's data
    circuit_overtake    circuit difficulty score built from this session's data

  All four simulators updated to use these fields.
  Public interface (run_prediction / detect_scenario_intent) unchanged.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.backend.config import settings
from src.backend.constants import DRIVER_CODES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# F1 Physics constants
# ─────────────────────────────────────────────────────────────────────────────

_PIT_LOSS: dict[str, float] = {
    "monaco":      22.5,
    "singapore":   23.0,
    "hungary":     20.5,
    "barcelona":   19.5,
    "spa":         20.0,
    "monza":       19.0,
    "silverstone": 20.5,
    "suzuka":      20.0,
    "abu dhabi":   20.5,
    "bahrain":     22.0,
    "jeddah":      21.0,
    "miami":       21.5,
    "austin":      21.0,
    "las vegas":   22.0,
    "default":     21.0,
}

_TYRE_DEG: dict[str, dict] = {
    "S": {"cliff_lap": 12, "deg_per_lap": 0.10, "pace_advantage_vs_m": -0.6},
    "M": {"cliff_lap": 25, "deg_per_lap": 0.06, "pace_advantage_vs_m":  0.0},
    "H": {"cliff_lap": 38, "deg_per_lap": 0.04, "pace_advantage_vs_m":  0.5},
    "I": {"cliff_lap": 20, "deg_per_lap": 0.07, "pace_advantage_vs_m":  0.3},
    "W": {"cliff_lap": 28, "deg_per_lap": 0.05, "pace_advantage_vs_m":  0.2},
    "U": {"cliff_lap": 20, "deg_per_lap": 0.07, "pace_advantage_vs_m":  0.0},
}

_SC_GAP_COMPRESSION_PER_LAP  = 28.0
_VSC_GAP_COMPRESSION_PER_LAP =  8.0
_SC_PIT_LOSS_FACTOR           = 0.55
_SC_TYPICAL_DURATION_LAPS     = 4

_DRS_OVERTAKE_THRESHOLD_S  = 1.0
_EASY_OVERTAKE_THRESHOLD_S = 0.5

# Throttle-cliff detection
_THROTTLE_PROTECTION_DROP_PCT  = 2.5   # % drop that signals protection mode
_THROTTLE_LOOKBACK_LAPS        = 6
_THROTTLE_MIN_LAPS_DROPPING    = 2     # consecutive declining laps to confirm


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_laps(session_id: str) -> list[dict]:
    path = settings.processed_dir / session_id / "laps.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _load_metadata(session_id: str) -> dict:
    path = settings.processed_dir / session_id / "metadata.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _pit_loss_for_track(track_name: str) -> float:
    t = (track_name or "").lower()
    for key, val in _PIT_LOSS.items():
        if key in t:
            return val
    return _PIT_LOSS["default"]


def _driver_name(code: str) -> str:
    return DRIVER_CODES.get(code, code)


# ─────────────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriverState:
    code: str
    name: str
    position: int
    gap_to_leader: float           # seconds behind leader
    gap_trend_3lap: float | None   # negative = closing on leader
    tyre: str                      # S / M / H / I / W / U
    tyre_age: int                  # laps on current compound
    tyre_deg_rate: float           # observed s/lap degradation
    last_lap_time: float
    avg_lap_time_5: float          # 5-lap rolling average
    speed_kph: float
    pit_stops_done: int
    # ── Tier 1 additions ──────────────────────────────────────────────────
    throttle_mode: str = "UNKNOWN"         # NORMAL | PROTECTING | UNKNOWN
    throttle_drop_pct: float | None = None # throttle% lost over lookback window
    max_speed_delta: float | None = None   # vs 5-lap baseline (neg = slower)


@dataclass
class RaceSnapshot:
    session_id: str
    race_name: str
    track_name: str
    current_lap: int
    total_laps: int
    laps_remaining: int
    pit_loss_s: float              # session-observed mean (or circuit default)
    drivers: list[DriverState] = field(default_factory=list)
    recent_laps: list[dict] = field(default_factory=list)
    # ── Tier 1 additions ──────────────────────────────────────────────────
    pit_stats: dict[str, Any] = field(default_factory=dict)
    circuit_overtake: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario_type: str
    trigger_lap: int
    laps_remaining: int
    affected_drivers: list[str]
    projected_positions: dict[str, int] = field(default_factory=dict)
    net_time_deltas: dict[str, float] = field(default_factory=dict)
    key_numbers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    assumptions: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — three new analytics functions
# ─────────────────────────────────────────────────────────────────────────────

def _detect_throttle_cliff(
    laps: list[dict],
    code: str,
    lookback: int = _THROTTLE_LOOKBACK_LAPS,
    threshold_pct: float = _THROTTLE_PROTECTION_DROP_PCT,
) -> dict[str, Any]:
    """
    Detect tyre protection mode from throttle_pct trend.

    A sustained throttle% drop > threshold_pct over lookback laps signals
    the driver is managing tyres — typically 2-3 laps BEFORE lap time reflects
    it, giving earlier warning than the current linear regression approach.

    Returns:
        mode               NORMAL | PROTECTING | UNKNOWN
        drop_pct           total throttle% lost from baseline to current
        laps_in_protection_mode
        confidence         0-1
        interpretation     human-readable string for the LLM
    """
    recent: list[float] = []
    for lap in reversed(laps[-lookback:]):
        t = (lap.get("throttle_pct") or {}).get(code)
        if t is not None:
            recent.append(float(t))
        if len(recent) >= lookback:
            break
    recent.reverse()   # oldest → newest

    if len(recent) < 3:
        return {
            "mode": "UNKNOWN",
            "throttle_baseline_pct": None,
            "throttle_current_pct": None,
            "drop_pct": None,
            "laps_in_protection_mode": 0,
            "confidence": 0.0,
            "interpretation": "Insufficient throttle data",
        }

    baseline = recent[0]
    current  = recent[-1]
    drop     = round(baseline - current, 2)

    # Count consecutive declining laps from the right
    laps_dropping = 0
    for i in range(len(recent) - 1, 0, -1):
        if recent[i] < recent[i - 1] - 0.3:
            laps_dropping += 1
        else:
            break

    in_protection = drop >= threshold_pct and laps_dropping >= _THROTTLE_MIN_LAPS_DROPPING
    mode = "PROTECTING" if in_protection else "NORMAL"

    confidence = 0.0
    if in_protection:
        confidence = min(0.95, round(
            laps_dropping / 6.0 + min(drop, 8.0) / 16.0, 2
        ))

    return {
        "mode": mode,
        "throttle_baseline_pct": round(baseline, 1),
        "throttle_current_pct": round(current, 1),
        "drop_pct": drop,
        "laps_in_protection_mode": laps_dropping,
        "confidence": confidence,
        "interpretation": (
            f"Throttle down {drop:.1f}% over {laps_dropping} laps — "
            "driver managing tyres, cliff likely 2-3 laps ahead"
            if mode == "PROTECTING"
            else "Throttle stable — no cliff signal yet"
        ),
    }


def _max_speed_delta(laps: list[dict], code: str, lookback: int = 5) -> float | None:
    """
    Compare the most recent lap's max_speed to the 5-lap rolling average.
    Negative = driver is slower than their recent baseline (lift-and-coast,
    drag damage, or ERS deployment change).
    """
    values: list[float] = []
    for lap in reversed(laps[-lookback:]):
        v = (lap.get("max_speed") or {}).get(code)
        if v and float(v) > 100:
            values.append(float(v))
    if len(values) < 2:
        return None
    current  = values[0]           # most recent (reversed order)
    baseline = sum(values[1:]) / len(values[1:])
    return round(current - baseline, 1)


def compute_pit_statistics(
    laps: list[dict],
    circuit_default_s: float,
) -> dict[str, Any]:
    """
    Build a stochastic pit model from actual pit_duration events in this session.

    pit_stops in laps.json contain:
        {"type": "pit_stop", "actor": "LEC",
         "details": {"pit_duration": 2.4, "compound": "S", "lap": 32}}

    Returns a dict with mean, std, p10/p90 total pit loss estimates
    so the simulators can quote confidence intervals instead of single values.
    """
    durations: list[float] = []
    for lap in laps:
        for ps in (lap.get("pit_stops") or []):
            if not isinstance(ps, dict):
                continue
            dur = (ps.get("details") or {}).get("pit_duration")
            if dur and float(dur) > 0.5:      # filter corrupt zeros
                durations.append(float(dur))

    if len(durations) < 2:
        return {
            "mean_stationary_s": None,
            "std_stationary_s": None,
            "total_pit_loss_mean_s": circuit_default_s,
            "total_pit_loss_p10_s":  round(circuit_default_s - 0.5, 2),
            "total_pit_loss_p90_s":  round(circuit_default_s + 1.2, 2),
            "sample_count": len(durations),
            "source": "circuit_default",
            "interpretation": (
                f"No stops recorded yet — using circuit default ({circuit_default_s}s)"
                if not durations
                else f"Only {len(durations)} stop observed — using circuit default"
            ),
        }

    mean_stat = round(statistics.mean(durations), 2)
    std_stat  = round(statistics.stdev(durations), 2)

    # Pit lane travel time ≈ total loss − stationary time.
    # We back-calculate travel time from the circuit default, assuming 2.5s
    # typical stationary time at that circuit.
    travel_time = round(circuit_default_s - 2.5, 2)

    total_mean  = round(travel_time + mean_stat, 2)
    total_p10   = round(travel_time + max(mean_stat - std_stat, mean_stat * 0.85), 2)
    total_p90   = round(travel_time + mean_stat + std_stat * 1.5, 2)

    return {
        "mean_stationary_s":    mean_stat,
        "std_stationary_s":     std_stat,
        "total_pit_loss_mean_s": total_mean,
        "total_pit_loss_p10_s":  total_p10,
        "total_pit_loss_p90_s":  total_p90,
        "sample_count": len(durations),
        "source": "session_observed",
        "interpretation": (
            f"Based on {len(durations)} stops this session: "
            f"expected {total_mean}s pit loss "
            f"(best {total_p10}s / worst {total_p90}s)"
        ),
    }


def compute_circuit_overtake_score(laps: list[dict]) -> dict[str, Any]:
    """
    Compute circuit-specific overtaking difficulty from real drs_pct and
    sector time spread observed in THIS session.

    Replaces the static 0.10 base probability in the overtake simulator with
    a score calibrated to the actual circuit and conditions today.

    Score 0-1: higher = easier to overtake.
      0.0 = Monaco-level (8% DRS, tight sectors)
      1.0 = Monza-level  (35% DRS, big sector spreads)
    """
    drs_vals:   list[float] = []
    s1_spreads: list[float] = []
    s3_spreads: list[float] = []

    for lap in laps[-10:]:       # last 10 laps = current track conditions
        drs_lap = lap.get("drs_pct") or {}
        for v in drs_lap.values():
            if isinstance(v, (int, float)) and v > 0:
                drs_vals.append(float(v))

        secs = lap.get("sector_times") or {}
        s1_times = [
            v[0] for v in secs.values()
            if v and len(v) >= 3 and isinstance(v[0], (int, float)) and v[0] > 5
        ]
        s3_times = [
            v[2] for v in secs.values()
            if v and len(v) >= 3 and isinstance(v[2], (int, float)) and v[2] > 5
        ]
        if len(s1_times) > 1:
            s1_spreads.append(max(s1_times) - min(s1_times))
        if len(s3_times) > 1:
            s3_spreads.append(max(s3_times) - min(s3_times))

    avg_drs_pct   = round(statistics.mean(drs_vals),   1) if drs_vals   else 20.0
    avg_s1_spread = round(statistics.mean(s1_spreads), 3) if s1_spreads else 0.3
    avg_s3_spread = round(statistics.mean(s3_spreads), 3) if s3_spreads else 0.3

    # DRS score: 8% → 0.0, 35% → 1.0
    drs_score    = min(1.0, max(0.0, (avg_drs_pct - 8.0) / 27.0))
    # Sector spread score: more gap variation = more attack opportunity
    sector_score = min(1.0, max(0.0, (avg_s1_spread + avg_s3_spread) / 2.5))
    combined     = round(drs_score * 0.65 + sector_score * 0.35, 3)

    difficulty = (
        "EASY"     if combined > 0.65 else
        "MODERATE" if combined > 0.35 else
        "HARD"
    )

    drs_label = (
        "long DRS zones (Monza-style)"    if avg_drs_pct > 30 else
        "medium DRS opportunity"           if avg_drs_pct > 18 else
        "very limited DRS (Monaco-style)"
    )

    return {
        "overtake_score":        combined,
        "difficulty":            difficulty,
        "avg_drs_pct":           avg_drs_pct,
        "avg_sector_spread_s":   round(avg_s1_spread + avg_s3_spread, 3),
        "drs_score_component":   round(drs_score, 3),
        "sector_score_component": round(sector_score, 3),
        "interpretation":        f"DRS active {avg_drs_pct}% of lap — {drs_label}",
        "source":                "session_data" if drs_vals else "defaults",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Existing analytics helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _observed_deg_rate(laps: list[dict], code: str, lookback: int = 5) -> float:
    """Linear regression over last N laps on the same compound."""
    relevant: list[tuple[float, str]] = []
    for lap in reversed(laps[-lookback:]):
        lt   = (lap.get("lap_times") or {}).get(code)
        tyre = (lap.get("tyres")     or {}).get(code)
        if lt and lt > 60.0:
            relevant.append((float(lt), str(tyre or "U")))
        if len(relevant) >= lookback:
            break

    relevant.reverse()
    if len(relevant) < 3:
        return 0.0

    base_tyre  = relevant[-1][1]
    same_stint = [lt for lt, t in relevant if t == base_tyre]
    if len(same_stint) < 2:
        return 0.0

    n      = len(same_stint)
    x_mean = (n - 1) / 2
    y_mean = sum(same_stint) / n
    num    = sum((i - x_mean) * (same_stint[i] - y_mean) for i in range(n))
    den    = sum((i - x_mean) ** 2 for i in range(n))
    return round(num / den, 4) if den != 0 else 0.0


def _rolling_avg_laptime(laps: list[dict], code: str, n: int = 5) -> float:
    times = []
    for lap in reversed(laps[-n:]):
        lt = (lap.get("lap_times") or {}).get(code)
        if lt and lt > 60.0:
            times.append(float(lt))
    return round(sum(times) / len(times), 3) if times else 0.0


def _gap_trend(laps: list[dict], code: str, lookback: int = 3) -> float | None:
    vals = [
        lap["gaps"].get(code)
        for lap in laps[-lookback:]
        if isinstance((lap.get("gaps") or {}).get(code), (int, float))
    ]
    if len(vals) < 2:
        return None
    return round(float(vals[-1]) - float(vals[0]), 3)


def _pit_stops_done(laps: list[dict], code: str) -> int:
    count = 0
    for lap in laps:
        for ps in (lap.get("pit_stops") or []):
            actor = ps.get("actor") if isinstance(ps, dict) else None
            if actor == code:
                count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder — now populates Tier 1 fields
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshot(
    session_id: str,
    live_context: dict[str, Any],
    laps_override: list[dict] | None = None,
) -> RaceSnapshot | None:
    laps = laps_override if laps_override is not None else _load_laps(session_id)
    if not laps:
        return None

    meta       = _load_metadata(session_id)
    track_name = meta.get("track_name") or live_context.get("race") or "Unknown"
    race_name  = f"{meta.get('year', '')} {meta.get('gp', live_context.get('race', ''))}".strip()

    current_lap    = int(live_context.get("current_lap") or laps[-1].get("lap", 0))
    total_laps     = int(live_context.get("total_laps") or meta.get("total_laps", 0))
    laps_remaining = max(0, total_laps - current_lap)

    circuit_default = _pit_loss_for_track(track_name)
    latest_lap      = laps[-1] if laps else {}

    # ── Tier 1: session-level calculations ────────────────────────────────
    pit_stats_data       = compute_pit_statistics(laps, circuit_default)
    circuit_overtake_data = compute_circuit_overtake_score(laps)

    # Use session-observed mean when we have enough data
    effective_pit_loss = (
        pit_stats_data["total_pit_loss_mean_s"]
        if pit_stats_data["source"] == "session_observed"
        else circuit_default
    )

    positions_raw = live_context.get("positions") or latest_lap.get("positions") or {}
    gaps_raw      = live_context.get("gaps")       or latest_lap.get("gaps")       or {}
    tyres_raw     = live_context.get("tyres")      or latest_lap.get("tyres")      or {}
    tyre_ages_raw = live_context.get("tyre_ages")  or latest_lap.get("tyre_ages")  or {}
    avg_speed_raw = live_context.get("avg_speed")  or latest_lap.get("avg_speed")  or {}

    drivers: list[DriverState] = []
    for code, pos in sorted(positions_raw.items(), key=lambda x: x[1]):
        tyre     = str(tyres_raw.get(code) or "U")
        tyre_age = int(tyre_ages_raw.get(code) or 1)

        deg_rate = _observed_deg_rate(laps, code)
        if deg_rate == 0.0:
            deg_rate = _TYRE_DEG.get(tyre, _TYRE_DEG["U"])["deg_per_lap"]

        last_lt = float((latest_lap.get("lap_times") or {}).get(code) or 0.0)
        avg_lt  = _rolling_avg_laptime(laps, code, n=5)

        # ── Tier 1: per-driver signals ─────────────────────────────────────
        throttle_data = _detect_throttle_cliff(laps, code)
        spd_delta     = _max_speed_delta(laps, code)

        drivers.append(DriverState(
            code=code,
            name=_driver_name(code),
            position=int(pos),
            gap_to_leader=float(gaps_raw.get(code) or 0.0),
            gap_trend_3lap=_gap_trend(laps, code),
            tyre=tyre,
            tyre_age=tyre_age,
            tyre_deg_rate=deg_rate,
            last_lap_time=last_lt,
            avg_lap_time_5=avg_lt,
            speed_kph=float(avg_speed_raw.get(code) or 0.0),
            pit_stops_done=_pit_stops_done(laps, code),
            throttle_mode=throttle_data["mode"],
            throttle_drop_pct=throttle_data["drop_pct"],
            max_speed_delta=spd_delta,
        ))

    return RaceSnapshot(
        session_id=session_id,
        race_name=race_name,
        track_name=track_name,
        current_lap=current_lap,
        total_laps=total_laps,
        laps_remaining=laps_remaining,
        pit_loss_s=effective_pit_loss,
        drivers=drivers,
        recent_laps=laps[-5:],
        pit_stats=pit_stats_data,
        circuit_overtake=circuit_overtake_data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator 1 — Safety Car / VSC
# ─────────────────────────────────────────────────────────────────────────────

def simulate_safety_car(
    snapshot: RaceSnapshot,
    sc_type: str = "SC",
    sc_duration_laps: int = _SC_TYPICAL_DURATION_LAPS,
) -> ScenarioResult:
    compression_per_lap = (
        _SC_GAP_COMPRESSION_PER_LAP if sc_type == "SC"
        else _VSC_GAP_COMPRESSION_PER_LAP
    )

    # Use session-observed pit loss distribution
    pit_loss_mean  = snapshot.pit_stats.get("total_pit_loss_mean_s",  snapshot.pit_loss_s)
    pit_loss_best  = snapshot.pit_stats.get("total_pit_loss_p10_s",   snapshot.pit_loss_s - 0.5)
    pit_loss_worst = snapshot.pit_stats.get("total_pit_loss_p90_s",   snapshot.pit_loss_s + 1.2)

    pit_loss_sc       = round(pit_loss_mean  * _SC_PIT_LOSS_FACTOR, 2)
    pit_loss_sc_best  = round(pit_loss_best  * _SC_PIT_LOSS_FACTOR, 2)
    pit_loss_sc_worst = round(pit_loss_worst * _SC_PIT_LOSS_FACTOR, 2)

    compressed_gaps: dict[str, float] = {
        d.code: round(max(0.0, d.gap_to_leader - compression_per_lap * sc_duration_laps), 2)
        for d in snapshot.drivers
    }

    should_pit:   list[str] = []
    pit_analyses: dict[str, dict] = {}

    for d in snapshot.drivers:
        if d.position > 10:
            continue
        tyre_params     = _TYRE_DEG.get(d.tyre, _TYRE_DEG["U"])
        laps_past_cliff = max(0, d.tyre_age - tyre_params["cliff_lap"])
        remaining_deg   = (
            d.tyre_deg_rate * snapshot.laps_remaining
            + laps_past_cliff * 0.03 * snapshot.laps_remaining
        )

        # Throttle protection = hidden deg not yet in lap time → inflate estimate
        throttle_extra = 0.0
        if d.throttle_mode == "PROTECTING" and d.throttle_drop_pct:
            throttle_extra = d.tyre_deg_rate * 3   # +3 laps worth of deg
            remaining_deg += throttle_extra

        net_benefit = remaining_deg - pit_loss_sc

        pit_analyses[d.code] = {
            "tyre":               d.tyre,
            "tyre_age":           d.tyre_age,
            "laps_past_cliff":    laps_past_cliff,
            "throttle_mode":      d.throttle_mode,
            "throttle_drop_pct":  d.throttle_drop_pct,
            "remaining_deg_s":    round(remaining_deg, 2),
            "pit_loss_sc_mean_s": pit_loss_sc,
            "net_benefit_s":      round(net_benefit, 2),
            "verdict": (
                "PIT" + (" (throttle protecting — urgent)" if d.throttle_mode == "PROTECTING" else "")
                if net_benefit > 0 else "STAY"
            ),
        }
        if net_benefit > 0:
            should_pit.append(d.code)

    effective_times = {
        d.code: compressed_gaps[d.code] + (pit_loss_sc if d.code in should_pit else 0.0)
        for d in snapshot.drivers
    }
    sorted_after        = sorted(effective_times.items(), key=lambda x: x[1])
    projected_positions = {code: i + 1 for i, (code, _) in enumerate(sorted_after)}
    net_deltas          = {
        d.code: round(effective_times[d.code] - d.gap_to_leader, 2)
        for d in snapshot.drivers
    }

    pos_changes = sorted(
        [
            {"driver": d.name,
             "from": d.position,
             "to": projected_positions.get(d.code, d.position),
             "delta": d.position - projected_positions.get(d.code, d.position)}
            for d in snapshot.drivers[:10]
            if d.position != projected_positions.get(d.code, d.position)
        ],
        key=lambda x: abs(x["delta"]),
        reverse=True,
    )

    return ScenarioResult(
        scenario_type=sc_type,
        trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=[d.code for d in snapshot.drivers[:10]],
        projected_positions=projected_positions,
        net_time_deltas=net_deltas,
        key_numbers={
            "sc_type":                    sc_type,
            "sc_duration_laps":           sc_duration_laps,
            "gap_compression_total_s":    round(compression_per_lap * sc_duration_laps, 1),
            "pit_loss_normal_mean_s":     round(pit_loss_mean, 1),
            "pit_loss_normal_range":      f"{round(pit_loss_best,1)}–{round(pit_loss_worst,1)}s",
            "pit_loss_under_sc_mean_s":   pit_loss_sc,
            "pit_loss_under_sc_best_s":   pit_loss_sc_best,
            "pit_data_source":            snapshot.pit_stats.get("source", "unknown"),
            "drivers_should_pit":         [_driver_name(c) for c in should_pit],
            "position_changes":           pos_changes[:6],
            "per_driver_analysis":        {
                _driver_name(c): v for c, v in list(pit_analyses.items())[:8]
            },
        },
        confidence=0.84,
        assumptions=[
            f"SC deployed for ~{sc_duration_laps} laps",
            f"Pit loss under {sc_type}: {pit_loss_sc}s mean "
            f"({snapshot.pit_stats.get('interpretation', 'circuit default')})",
            "Throttle protection mode increases pit urgency score",
            "Gap compression: linear per SC lap",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator 2 — Pit stop
# ─────────────────────────────────────────────────────────────────────────────

def simulate_pit_stop(
    snapshot: RaceSnapshot,
    driver_code: str,
    pit_lap: int | None = None,
    new_compound: str | None = None,
) -> ScenarioResult:
    driver = next(
        (d for d in snapshot.drivers if d.code.upper() == driver_code.upper()), None
    )
    if driver is None:
        return ScenarioResult(
            scenario_type="pit_stop",
            trigger_lap=snapshot.current_lap,
            laps_remaining=snapshot.laps_remaining,
            affected_drivers=[driver_code],
            confidence=0.0,
            assumptions=[f"Driver {driver_code} not found in race data"],
        )

    pit_lap        = pit_lap or snapshot.current_lap
    laps_after_pit = max(1, snapshot.total_laps - pit_lap)

    if new_compound is None:
        new_compound = "M" if driver.tyre == "S" else ("H" if driver.tyre == "M" else "M")

    tyre_params     = _TYRE_DEG.get(driver.tyre,  _TYRE_DEG["U"])
    new_tyre_params = _TYRE_DEG.get(new_compound, _TYRE_DEG["M"])

    # Session-observed pit loss with range
    tc_mean  = snapshot.pit_stats.get("total_pit_loss_mean_s", snapshot.pit_loss_s)
    tc_best  = snapshot.pit_stats.get("total_pit_loss_p10_s",  snapshot.pit_loss_s - 0.5)
    tc_worst = snapshot.pit_stats.get("total_pit_loss_p90_s",  snapshot.pit_loss_s + 1.2)

    laps_past_cliff = max(0, driver.tyre_age - tyre_params["cliff_lap"])

    # Throttle protection → inflate effective deg rate
    effective_deg_rate = driver.tyre_deg_rate
    if driver.throttle_mode == "PROTECTING" and driver.throttle_drop_pct:
        effective_deg_rate += driver.throttle_drop_pct * 0.04

    current_deg       = effective_deg_rate * laps_after_pit + laps_past_cliff * 0.05 * laps_after_pit
    new_deg           = new_tyre_params["deg_per_lap"] * laps_after_pit
    pace_delta_per_lap = new_tyre_params["pace_advantage_vs_m"] - tyre_params.get("pace_advantage_vs_m", 0.0)
    pace_gain          = pace_delta_per_lap * laps_after_pit * -1

    tyre_benefit   = current_deg - new_deg + pace_gain
    net_mean       = round(tc_mean  - tyre_benefit, 2)
    net_best       = round(tc_best  - tyre_benefit, 2)
    net_worst      = round(tc_worst - tyre_benefit, 2)

    d_ahead  = next((d for d in snapshot.drivers if d.position == driver.position - 1), None)
    d_behind = next((d for d in snapshot.drivers if d.position == driver.position + 1), None)

    gap_ahead  = round(driver.gap_to_leader - d_ahead.gap_to_leader,  2) if d_ahead  else None
    gap_behind = round(d_behind.gap_to_leader - driver.gap_to_leader, 2) if d_behind else None

    rejoin_gap = driver.gap_to_leader + tc_mean
    rejoin_pos = sum(
        1 for d in snapshot.drivers
        if d.code != driver.code and d.gap_to_leader < rejoin_gap
    ) + 1

    laps_to_recover = None
    if net_mean < 0 and abs(pace_delta_per_lap) > 0:
        laps_to_recover = round(abs(net_mean) / abs(pace_delta_per_lap), 1)

    undercut_threat = None
    if d_behind and gap_behind is not None and gap_behind < 3.0:
        undercut_threat = {
            "from_driver":          d_behind.name,
            "from_driver_tyre":     f"{d_behind.tyre} age {d_behind.tyre_age}",
            "current_gap_behind_s": gap_behind,
            "threat_level":         "HIGH" if gap_behind < 1.5 else "MEDIUM",
            "note":                 "Could pit next lap and emerge ahead on fresh rubber",
        }

    return ScenarioResult(
        scenario_type="pit_stop",
        trigger_lap=pit_lap,
        laps_remaining=laps_after_pit,
        affected_drivers=[driver.code] + ([d_behind.code] if d_behind else []),
        projected_positions={driver.code: rejoin_pos},
        net_time_deltas={driver.code: net_mean},
        key_numbers={
            "driver":                      driver.name,
            "current_position":            driver.position,
            "pit_at_lap":                  pit_lap,
            "laps_remaining_after_pit":    laps_after_pit,
            "current_tyre":                driver.tyre,
            "current_tyre_age_laps":       driver.tyre_age,
            "laps_past_cliff":             laps_past_cliff,
            "throttle_mode":               driver.throttle_mode,
            "throttle_drop_pct":           driver.throttle_drop_pct,
            "effective_deg_rate_s_per_lap": round(effective_deg_rate, 3),
            "new_compound":                new_compound,
            "pit_loss_mean_s":             round(tc_mean, 2),
            "pit_loss_range":              f"{round(tc_best,1)}–{round(tc_worst,1)}s",
            "pit_data_source":             snapshot.pit_stats.get("source", "unknown"),
            "tyre_benefit_s":              round(tyre_benefit, 2),
            "net_time_mean_s":             net_mean,
            "net_time_best_s":             net_best,
            "net_time_worst_s":            net_worst,
            "rejoin_position":             rejoin_pos,
            "gap_to_car_ahead_s":          gap_ahead,
            "gap_to_car_behind_s":         gap_behind,
            "laps_to_recover_position":    laps_to_recover,
            "undercut_threat":             undercut_threat,
            "car_ahead":                   d_ahead.name  if d_ahead  else None,
            "car_behind":                  d_behind.name if d_behind else None,
            "verdict": (
                "PIT NOW — tyre benefit outweighs pit loss"
                + (" (throttle protection confirms urgency)" if driver.throttle_mode == "PROTECTING" else "")
                if net_mean < 0
                else "HOLD — pit loss exceeds tyre gain; consider overcut or wait for SC"
            ),
        },
        confidence=0.82,
        assumptions=[
            f"Pit loss: {round(tc_mean,1)}s mean (range {round(tc_best,1)}–{round(tc_worst,1)}s) "
            f"from {snapshot.pit_stats.get('source', 'circuit default')}",
            f"New compound: {new_compound}, deg: {new_tyre_params['deg_per_lap']}s/lap",
            f"Observed deg: {driver.tyre_deg_rate:.3f}s/lap"
            + (f" → adjusted to {effective_deg_rate:.3f} (throttle protection active)"
               if driver.throttle_mode == "PROTECTING" else ""),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator 3 — Tyre degradation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_tyre_degradation(
    snapshot: RaceSnapshot,
    driver_codes: list[str] | None = None,
) -> ScenarioResult:
    codes       = driver_codes or [d.code for d in snapshot.drivers[:6]]
    projections: dict[str, dict] = {}
    cliff_warnings: list[str]    = []

    for d in snapshot.drivers:
        if d.code not in codes:
            continue

        tyre_params     = _TYRE_DEG.get(d.tyre, _TYRE_DEG["U"])
        laps_to_cliff   = max(0, tyre_params["cliff_lap"] - d.tyre_age)
        laps_past_cliff = max(0, d.tyre_age - tyre_params["cliff_lap"])

        base_time = d.avg_lap_time_5 if d.avg_lap_time_5 > 60 else d.last_lap_time
        if base_time == 0:
            continue

        # Throttle protection: inflate effective deg rate
        effective_deg = d.tyre_deg_rate
        if d.throttle_mode == "PROTECTING" and d.throttle_drop_pct:
            effective_deg += d.throttle_drop_pct * 0.04

        lap_projections = []
        for fl in range(1, min(snapshot.laps_remaining + 1, 16)):
            fa          = d.tyre_age + fl
            deg_penalty = effective_deg * fl
            cliff_extra = max(0, fa - tyre_params["cliff_lap"]) * 0.03
            lap_projections.append(round(base_time + deg_penalty + cliff_extra, 3))

        total_deg_loss = round(
            sum(lap_projections) - base_time * len(lap_projections), 2
        ) if lap_projections else 0.0

        throttle_warn = d.throttle_mode == "PROTECTING"
        status = (
            "CRITICAL" if laps_past_cliff > 5 else
            "WARNING"  if laps_past_cliff > 0 or laps_to_cliff <= 3 or throttle_warn else
            "OK"
        )

        projections[d.name] = {
            "position":                    d.position,
            "tyre":                        d.tyre,
            "tyre_age_now":               d.tyre_age,
            "cliff_at_lap":               tyre_params["cliff_lap"],
            "laps_to_cliff":              laps_to_cliff,
            "laps_past_cliff":            laps_past_cliff,
            "throttle_mode":              d.throttle_mode,
            "throttle_drop_pct":          d.throttle_drop_pct,
            "observed_deg_s_per_lap":     round(d.tyre_deg_rate, 3),
            "effective_deg_s_per_lap":    round(effective_deg, 3),
            "projected_lap_times_next_5": lap_projections[:5],
            "total_deg_loss_remaining_s": total_deg_loss,
            "status":                     status,
        }

        if status in ("CRITICAL", "WARNING"):
            extra = f" + throttle down {d.throttle_drop_pct:.1f}%" if throttle_warn else ""
            cliff_warnings.append(
                f"{d.name} P{d.position} on {d.tyre} age {d.tyre_age} — {status}{extra}"
            )

    return ScenarioResult(
        scenario_type="tyre_degradation",
        trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=codes,
        key_numbers={
            "driver_projections": projections,
            "cliff_warnings":     cliff_warnings,
            "laps_remaining":     snapshot.laps_remaining,
        },
        confidence=0.80,
        assumptions=[
            "Deg rate from linear regression over last 5 laps on same compound",
            "Throttle protection signal adds 0.04s/lap per 1% throttle drop to effective rate",
            "Cliff steepness adds 0.03s/lap per lap beyond the nominal cliff",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator 4 — Overtake window
# ─────────────────────────────────────────────────────────────────────────────

def simulate_overtake_window(
    snapshot: RaceSnapshot,
    attacker_code: str,
    defender_code: str,
) -> ScenarioResult:
    attacker = next((d for d in snapshot.drivers if d.code.upper() == attacker_code.upper()), None)
    defender = next((d for d in snapshot.drivers if d.code.upper() == defender_code.upper()), None)

    if not attacker or not defender:
        return ScenarioResult(
            scenario_type="overtake_window",
            trigger_lap=snapshot.current_lap,
            laps_remaining=snapshot.laps_remaining,
            affected_drivers=[attacker_code, defender_code],
            confidence=0.0,
            assumptions=["One or both drivers not found in race data"],
        )

    current_gap = round(abs(attacker.gap_to_leader - defender.gap_to_leader), 3)
    att_lt      = attacker.avg_lap_time_5 or attacker.last_lap_time
    def_lt      = defender.avg_lap_time_5 or defender.last_lap_time
    pace_delta  = round(att_lt - def_lt, 3)   # negative = attacker faster

    laps_to_close = (
        round(current_gap / abs(pace_delta), 1)
        if pace_delta < 0 and current_gap > 0 else None
    )

    drs_eligible = current_gap <= _DRS_OVERTAKE_THRESHOLD_S
    easy_drs     = current_gap <= _EASY_OVERTAKE_THRESHOLD_S

    tyre_advantage = round(
        defender.tyre_deg_rate * snapshot.laps_remaining
        - attacker.tyre_deg_rate * snapshot.laps_remaining,
        2,
    )

    # ── Tier 1: base probability from circuit overtake score ──────────────
    circuit_score = snapshot.circuit_overtake.get("overtake_score", 0.5)
    circuit_diff  = snapshot.circuit_overtake.get("difficulty", "MODERATE")

    # Base: 0.05 (hard circuit) to 0.30 (easy circuit)
    p = round(0.05 + circuit_score * 0.25, 3)

    if pace_delta < -0.3:                                           p += 0.30
    if pace_delta < -0.6:                                           p += 0.10
    if drs_eligible:                                                p += 0.25
    if easy_drs:                                                    p += 0.15
    if tyre_advantage > 2.0:                                        p += 0.20
    if attacker.gap_trend_3lap and attacker.gap_trend_3lap < -0.5: p += 0.10
    if attacker.tyre == "S" and defender.tyre == "H" and attacker.tyre_age < 15:
        p += 0.10  # fresh soft vs worn hard

    # Defender throttle protecting = their tyres are degrading = easier to pass
    if defender.throttle_mode == "PROTECTING":                      p += 0.12

    # Attacker straight-line speed advantage
    if attacker.max_speed_delta and attacker.max_speed_delta > 3.0: p += 0.05

    p = min(0.95, round(p, 2))

    verdict = (
        "HIGHLY LIKELY" if p > 0.70 else
        "LIKELY"        if p > 0.50 else
        "POSSIBLE"      if p > 0.30 else
        "UNLIKELY"
    )

    return ScenarioResult(
        scenario_type="overtake_window",
        trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=[attacker_code, defender_code],
        key_numbers={
            "attacker":                      attacker.name,
            "defender":                      defender.name,
            "attacker_position":             attacker.position,
            "defender_position":             defender.position,
            "current_gap_s":                 current_gap,
            "pace_delta_s_per_lap":          pace_delta,
            "laps_to_close_at_current_pace": laps_to_close,
            "drs_eligible":                  drs_eligible,
            "circuit_difficulty":            circuit_diff,
            "circuit_overtake_score":        circuit_score,
            "circuit_drs_pct":               snapshot.circuit_overtake.get("avg_drs_pct"),
            "tyre_advantage_remaining_s":    tyre_advantage,
            "attacker_tyre":                 f"{attacker.tyre} age {attacker.tyre_age}",
            "defender_tyre":                 f"{defender.tyre} age {defender.tyre_age}",
            "defender_throttle_mode":        defender.throttle_mode,
            "attacker_max_speed_delta":      attacker.max_speed_delta,
            "gap_trend_3lap_s":              attacker.gap_trend_3lap,
            "overtake_probability":          p,
            "verdict":                       verdict,
        },
        confidence=0.76,
        assumptions=[
            f"Circuit difficulty: {circuit_diff} (score {circuit_score} from session drs_pct + sectors)",
            "Defender in throttle protection adds +12% to attacker probability",
            "Pace delta from 5-lap rolling average",
            "DRS effectiveness: ~0.3–0.5s per lap",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_KEYWORDS: dict[str, list[str]] = {
    "safety_car": [
        "safety car", "sc comes in", "sc deployed", "safety car out",
        "vsc", "virtual safety car", "neutralization", "yellow flag",
        "full course", "sc out",
    ],
    "pit_stop": [
        "pit", "box", "pits at", "stops at", "pitstop", "pit window",
        "undercut", "overcut", "pit at lap", "stop now",
        "come in", "bring him in",
    ],
    "tyre_degradation": [
        "tyre deg", "tire deg", "tyre wear", "tire wear",
        "struggling with tyres", "struggling with tires",
        "tyre life", "cliff", "worn tyres", "worn tires",
        "how long can", "surviving on", "tire life",
    ],
    "overtake_window": [
        "overtake", "pass", "gap closing", "gap is closing", "catch", "drs",
        "attack", "fight for position", "battle", "can he catch",
        "can she catch", "behind", "chasing",
    ],
}


def detect_scenario_intent(query: str) -> str:
    q      = query.lower()
    scores = {k: 0 for k in _SCENARIO_KEYWORDS}
    for scenario, keywords in _SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[scenario] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "pit_stop"


def _extract_driver_codes_from_query(query: str, snapshot: RaceSnapshot) -> list[str]:
    q_upper = query.upper()
    q_lower = query.lower()
    return [
        d.code for d in snapshot.drivers
        if d.code in q_upper or d.name.lower() in q_lower
    ]


def _extract_lap_number(query: str) -> int | None:
    m = re.search(r'\blap\s+(\d+)\b', query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'\bat\s+(\d+)\b', query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (public interface unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(
    query: str,
    session_id: str,
    live_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the appropriate scenario simulation for a natural language query.

    Returns:
        snapshot      serialised RaceSnapshot (top-10 driver states + Tier 1 data)
        scenario      serialised ScenarioResult (computed numbers)
        scenario_type str
        error         str | None
    """
    try:
        snapshot = build_snapshot(session_id, live_context)
        if snapshot is None or not snapshot.drivers:
            return {
                "snapshot":      None,
                "scenario":      None,
                "scenario_type": "unknown",
                "error":         "No race data available — laps.json is empty or missing.",
            }

        intent    = detect_scenario_intent(query)
        mentioned = _extract_driver_codes_from_query(query, snapshot)
        pit_lap   = _extract_lap_number(query)

        if intent == "safety_car":
            sc_type = "VSC" if any(k in query.lower() for k in ("vsc", "virtual")) else "SC"
            result  = simulate_safety_car(snapshot, sc_type=sc_type)

        elif intent == "tyre_degradation":
            codes  = mentioned[:4] if mentioned else [d.code for d in snapshot.drivers[:6]]
            result = simulate_tyre_degradation(snapshot, driver_codes=codes)

        elif intent == "overtake_window":
            if len(mentioned) >= 2:
                attacker, defender = mentioned[0], mentioned[1]
            elif len(mentioned) == 1:
                d = next((dr for dr in snapshot.drivers if dr.code == mentioned[0]), None)
                if d and d.position > 1:
                    d_ahead  = next((dr for dr in snapshot.drivers if dr.position == d.position - 1), None)
                    attacker = mentioned[0]
                    defender = d_ahead.code if d_ahead else snapshot.drivers[0].code
                else:
                    attacker = snapshot.drivers[1].code if len(snapshot.drivers) > 1 else mentioned[0]
                    defender = snapshot.drivers[0].code
            else:
                attacker = snapshot.drivers[1].code if len(snapshot.drivers) > 1 else ""
                defender = snapshot.drivers[0].code
            result = simulate_overtake_window(snapshot, attacker, defender)

        else:   # pit_stop (default)
            driver_code = mentioned[0] if mentioned else next(
                (d.code for d in snapshot.drivers if d.position == 1),
                snapshot.drivers[0].code,
            )
            result = simulate_pit_stop(snapshot, driver_code=driver_code, pit_lap=pit_lap)

        return {
            "snapshot": {
                "race_name":        snapshot.race_name,
                "track_name":       snapshot.track_name,
                "current_lap":      snapshot.current_lap,
                "total_laps":       snapshot.total_laps,
                "laps_remaining":   snapshot.laps_remaining,
                "pit_loss_s":       snapshot.pit_loss_s,
                "pit_stats":        snapshot.pit_stats,
                "circuit_overtake": snapshot.circuit_overtake,
                "top10_drivers": [
                    {
                        "pos":              d.position,
                        "code":             d.code,
                        "name":             d.name,
                        "gap_s":            d.gap_to_leader,
                        "gap_trend_3lap_s": d.gap_trend_3lap,
                        "tyre":             d.tyre,
                        "tyre_age":         d.tyre_age,
                        "deg_rate":         d.tyre_deg_rate,
                        "avg_laptime_5":    d.avg_lap_time_5,
                        "pits_done":        d.pit_stops_done,
                        "throttle_mode":    d.throttle_mode,
                        "throttle_drop_pct": d.throttle_drop_pct,
                        "max_speed_delta":  d.max_speed_delta,
                    }
                    for d in snapshot.drivers[:10]
                ],
            },
            "scenario":      asdict(result),
            "scenario_type": intent,
            "error":         None,
        }

    except Exception as exc:
        logger.exception("Prediction engine error: %s", exc)
        return {
            "snapshot":      None,
            "scenario":      None,
            "scenario_type": "unknown",
            "error":         str(exc),
        }
