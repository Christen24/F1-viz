"""
F1 Race Prediction Engine  —  Tier 2 upgrade
=============================================

New in this version:
  simulate_race_to_finish()     Lap-by-lap projection of every driver to the
                                 chequered flag. Applies deg, cliff steepening,
                                 fuel-burn gain, and automatic pit decisions
                                 each simulated lap.

  simulate_strategy_comparison() Runs simulate_race_to_finish() twice (or more)
                                 under different strategy plans and returns the
                                 delta between them — the core of what F1 pit
                                 walls actually compute in real time.

  Two new dataclasses:
    DriverSimState               Per-driver mutable state inside the simulator
    RaceToFinishResult           Output of simulate_race_to_finish()

  run_prediction() extended with two new intents:
    "race_projection"            Projects full race to finish
    "strategy_comparison"        Compares two or more pit-stop strategies

  All existing Tier 1 / v1 code is preserved unchanged.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
from copy import deepcopy
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

# Fuel load parameters
_FUEL_GAIN_PER_LAP_S = 0.035   # s/lap gained as fuel burns off (~0.035s per kg)

# Tier 1 throttle detection
_THROTTLE_PROTECTION_DROP_PCT = 2.5
_THROTTLE_LOOKBACK_LAPS       = 6
_THROTTLE_MIN_LAPS_DROPPING   = 2


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
    gap_to_leader: float
    gap_trend_3lap: float | None
    tyre: str
    tyre_age: int
    tyre_deg_rate: float
    last_lap_time: float
    avg_lap_time_5: float
    speed_kph: float
    pit_stops_done: int
    # Tier 1
    throttle_mode: str = "UNKNOWN"
    throttle_drop_pct: float | None = None
    max_speed_delta: float | None = None


@dataclass
class RaceSnapshot:
    session_id: str
    race_name: str
    track_name: str
    current_lap: int
    total_laps: int
    laps_remaining: int
    pit_loss_s: float
    drivers: list[DriverState] = field(default_factory=list)
    recent_laps: list[dict] = field(default_factory=list)
    # Tier 1
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
# Tier 2 — new data structures for race simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriverSimState:
    """
    Mutable per-driver state inside simulate_race_to_finish().
    Copied from DriverState at simulation start; mutated each simulated lap.
    """
    code: str
    name: str
    cum_time: float          # cumulative race time (gap at start + all lap times)
    tyre: str
    tyre_age: int
    base_lt: float           # clean-air reference lap time at start of sim
    deg_rate: float          # observed s/lap degradation
    pits_done: int
    pit_laps: list[int] = field(default_factory=list)
    lap_times_sim: list[float] = field(default_factory=list)  # per-lap in simulation
    cum_series: list[float] = field(default_factory=list)     # cumulative time series


@dataclass
class RaceToFinishResult:
    """
    Output of simulate_race_to_finish().
    Contains the projected final order, all pit decisions made,
    gap series, and the key delta numbers for the LLM.
    """
    from_lap: int
    total_laps: int
    laps_simulated: int
    # Final order: list of dicts sorted P1→Pn
    final_order: list[dict] = field(default_factory=list)
    # Gap series: {driver_code: [gap_to_leader per simulated lap]}
    gap_series: dict[str, list[float]] = field(default_factory=dict)
    # Pit events: [{lap, driver, compound, type="auto"|"forced"}]
    pit_events: list[dict] = field(default_factory=list)
    # Key summary numbers for the LLM
    key_numbers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.78
    assumptions: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 analytics functions (unchanged from Tier 1 upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_throttle_cliff(
    laps: list[dict],
    code: str,
    lookback: int = _THROTTLE_LOOKBACK_LAPS,
    threshold_pct: float = _THROTTLE_PROTECTION_DROP_PCT,
) -> dict[str, Any]:
    recent: list[float] = []
    for lap in reversed(laps[-lookback:]):
        t = (lap.get("throttle_pct") or {}).get(code)
        if t is not None:
            recent.append(float(t))
        if len(recent) >= lookback:
            break
    recent.reverse()

    if len(recent) < 3:
        return {"mode": "UNKNOWN", "throttle_baseline_pct": None,
                "throttle_current_pct": None, "drop_pct": None,
                "laps_in_protection_mode": 0, "confidence": 0.0,
                "interpretation": "Insufficient throttle data"}

    baseline = recent[0]
    current  = recent[-1]
    drop     = round(baseline - current, 2)

    laps_dropping = 0
    for i in range(len(recent) - 1, 0, -1):
        if recent[i] < recent[i - 1] - 0.3:
            laps_dropping += 1
        else:
            break

    in_protection = drop >= threshold_pct and laps_dropping >= _THROTTLE_MIN_LAPS_DROPPING
    mode = "PROTECTING" if in_protection else "NORMAL"
    confidence = min(0.95, round(laps_dropping / 6.0 + min(drop, 8.0) / 16.0, 2)) if in_protection else 0.0

    return {
        "mode": mode,
        "throttle_baseline_pct": round(baseline, 1),
        "throttle_current_pct":  round(current, 1),
        "drop_pct":              drop,
        "laps_in_protection_mode": laps_dropping,
        "confidence":            confidence,
        "interpretation": (
            f"Throttle down {drop:.1f}% over {laps_dropping} laps — cliff likely 2-3 laps ahead"
            if mode == "PROTECTING" else "Throttle stable — no cliff signal yet"
        ),
    }


def _max_speed_delta(laps: list[dict], code: str, lookback: int = 5) -> float | None:
    values: list[float] = []
    for lap in reversed(laps[-lookback:]):
        v = (lap.get("max_speed") or {}).get(code)
        if v and float(v) > 100:
            values.append(float(v))
    if len(values) < 2:
        return None
    return round(values[0] - sum(values[1:]) / len(values[1:]), 1)


def compute_pit_statistics(laps: list[dict], circuit_default_s: float) -> dict[str, Any]:
    durations: list[float] = []
    for lap in laps:
        for ps in (lap.get("pit_stops") or []):
            if not isinstance(ps, dict):
                continue
            dur = (ps.get("details") or {}).get("pit_duration")
            if dur and float(dur) > 0.5:
                durations.append(float(dur))

    if len(durations) < 2:
        return {
            "mean_stationary_s": None, "std_stationary_s": None,
            "total_pit_loss_mean_s": circuit_default_s,
            "total_pit_loss_p10_s":  round(circuit_default_s - 0.5, 2),
            "total_pit_loss_p90_s":  round(circuit_default_s + 1.2, 2),
            "sample_count": len(durations), "source": "circuit_default",
            "interpretation": f"No stops recorded yet — using circuit default ({circuit_default_s}s)",
        }

    mean_stat = round(statistics.mean(durations), 2)
    std_stat  = round(statistics.stdev(durations), 2)
    travel    = round(circuit_default_s - 2.5, 2)
    return {
        "mean_stationary_s":     mean_stat,
        "std_stationary_s":      std_stat,
        "total_pit_loss_mean_s": round(travel + mean_stat, 2),
        "total_pit_loss_p10_s":  round(travel + max(mean_stat - std_stat, mean_stat * 0.85), 2),
        "total_pit_loss_p90_s":  round(travel + mean_stat + std_stat * 1.5, 2),
        "sample_count": len(durations), "source": "session_observed",
        "interpretation": (
            f"Based on {len(durations)} stops: "
            f"expected {round(travel+mean_stat,2)}s pit loss"
        ),
    }


def compute_circuit_overtake_score(laps: list[dict]) -> dict[str, Any]:
    drs_vals: list[float] = []
    s1_spreads: list[float] = []
    s3_spreads: list[float] = []

    for lap in laps[-10:]:
        for v in (lap.get("drs_pct") or {}).values():
            if isinstance(v, (int, float)) and v > 0:
                drs_vals.append(float(v))
        secs = lap.get("sector_times") or {}
        s1 = [v[0] for v in secs.values() if v and len(v) >= 3 and isinstance(v[0], (int, float)) and v[0] > 5]
        s3 = [v[2] for v in secs.values() if v and len(v) >= 3 and isinstance(v[2], (int, float)) and v[2] > 5]
        if len(s1) > 1: s1_spreads.append(max(s1) - min(s1))
        if len(s3) > 1: s3_spreads.append(max(s3) - min(s3))

    avg_drs = round(statistics.mean(drs_vals), 1) if drs_vals else 20.0
    avg_s1  = round(statistics.mean(s1_spreads), 3) if s1_spreads else 0.3
    avg_s3  = round(statistics.mean(s3_spreads), 3) if s3_spreads else 0.3
    drs_score    = min(1.0, max(0.0, (avg_drs - 8.0) / 27.0))
    sector_score = min(1.0, max(0.0, (avg_s1 + avg_s3) / 2.5))
    combined = round(drs_score * 0.65 + sector_score * 0.35, 3)

    return {
        "overtake_score": combined,
        "difficulty": "EASY" if combined > 0.65 else "MODERATE" if combined > 0.35 else "HARD",
        "avg_drs_pct": avg_drs,
        "avg_sector_spread_s": round(avg_s1 + avg_s3, 3),
        "drs_score_component": round(drs_score, 3),
        "sector_score_component": round(sector_score, 3),
        "interpretation": f"DRS active {avg_drs}% of lap",
        "source": "session_data" if drs_vals else "defaults",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Existing analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _observed_deg_rate(laps: list[dict], code: str, lookback: int = 5) -> float:
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
            if isinstance(ps, dict) and ps.get("actor") == code:
                count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder
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

    pit_stats_data        = compute_pit_statistics(laps, circuit_default)
    circuit_overtake_data = compute_circuit_overtake_score(laps)
    effective_pit_loss    = (
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
        last_lt       = float((latest_lap.get("lap_times") or {}).get(code) or 0.0)
        avg_lt        = _rolling_avg_laptime(laps, code, n=5)
        throttle_data = _detect_throttle_cliff(laps, code)
        spd_delta     = _max_speed_delta(laps, code)

        drivers.append(DriverState(
            code=code, name=_driver_name(code), position=int(pos),
            gap_to_leader=float(gaps_raw.get(code) or 0.0),
            gap_trend_3lap=_gap_trend(laps, code),
            tyre=tyre, tyre_age=tyre_age, tyre_deg_rate=deg_rate,
            last_lap_time=last_lt, avg_lap_time_5=avg_lt,
            speed_kph=float(avg_speed_raw.get(code) or 0.0),
            pit_stops_done=_pit_stops_done(laps, code),
            throttle_mode=throttle_data["mode"],
            throttle_drop_pct=throttle_data["drop_pct"],
            max_speed_delta=spd_delta,
        ))

    return RaceSnapshot(
        session_id=session_id, race_name=race_name, track_name=track_name,
        current_lap=current_lap, total_laps=total_laps, laps_remaining=laps_remaining,
        pit_loss_s=effective_pit_loss, drivers=drivers, recent_laps=laps[-5:],
        pit_stats=pit_stats_data, circuit_overtake=circuit_overtake_data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Race simulator internals
# ─────────────────────────────────────────────────────────────────────────────

def _best_new_compound(current_tyre: str, pits_done: int) -> str:
    """Pick the most logical next compound given current tyre and stop count."""
    if current_tyre == "S":
        return "M"
    if current_tyre == "M":
        return "H" if pits_done >= 1 else "M"
    return "M"   # H or unknown → medium


def _auto_pit_check(
    tyre: str,
    tyre_age: int,
    laps_left: int,
    pit_loss: float,
    deg_rate: float,
    new_compound: str = "M",
) -> bool:
    """
    Return True if stopping NOW is net beneficial over remaining laps.
    Compares remaining deg cost on current tyre vs pit_loss + deg cost on new tyre.
    Never pits with ≤4 laps left.
    """
    if laps_left <= 4:
        return False
    params      = _TYRE_DEG.get(tyre, _TYRE_DEG["U"])
    new_params  = _TYRE_DEG.get(new_compound, _TYRE_DEG["M"])
    laps_past   = max(0, tyre_age - params["cliff_lap"])
    current_cost = deg_rate * laps_left + laps_past * 0.04 * laps_left
    new_cost     = new_params["deg_per_lap"] * laps_left
    pace_delta   = new_params["pace_advantage_vs_m"] - params.get("pace_advantage_vs_m", 0.0)
    pace_gain    = pace_delta * laps_left * -1
    tyre_benefit = current_cost - new_cost + pace_gain
    return tyre_benefit > pit_loss


def _run_simulation(
    snapshot: RaceSnapshot,
    strategy_overrides: dict[str, list[int]] | None = None,
    max_stops: int = 2,
    fuel_gain_per_lap: float = _FUEL_GAIN_PER_LAP_S,
) -> dict[str, DriverSimState]:
    """
    Core lap-by-lap simulation engine.
    Returns dict {driver_code: DriverSimState} after simulating all remaining laps.

    strategy_overrides: {code: [abs_lap_numbers_to_pit]}
        e.g. {"LEC": [45]} forces LEC to pit at lap 45
        Auto-pit decisions are still made for all other laps/drivers.
    """
    overrides    = strategy_overrides or {}
    laps_to_sim  = snapshot.laps_remaining
    pit_loss     = snapshot.pit_loss_s

    # Initialise sim state from snapshot
    state: dict[str, DriverSimState] = {}
    for d in snapshot.drivers:
        base_lt = d.avg_lap_time_5 if d.avg_lap_time_5 > 60 else d.last_lap_time
        if base_lt == 0:
            base_lt = 92.0   # fallback
        state[d.code] = DriverSimState(
            code=d.code, name=d.name,
            cum_time=d.gap_to_leader,
            tyre=d.tyre, tyre_age=d.tyre_age,
            base_lt=base_lt, deg_rate=d.tyre_deg_rate,
            pits_done=d.pit_stops_done,
        )

    for sim_lap in range(1, laps_to_sim + 1):
        abs_lap        = snapshot.current_lap + sim_lap
        laps_left_after = laps_to_sim - sim_lap

        for code, s in state.items():
            params        = _TYRE_DEG.get(s.tyre, _TYRE_DEG["U"])
            age_this_lap  = s.tyre_age + 1
            laps_past     = max(0, age_this_lap - params["cliff_lap"])

            # Lap time components
            deg_penalty = s.deg_rate * age_this_lap
            cliff_extra = laps_past * 0.03
            fuel_bonus  = sim_lap * fuel_gain_per_lap
            lap_time    = s.base_lt + deg_penalty + cliff_extra - fuel_bonus

            # Pit decision
            pit_this_lap = False
            pit_type     = ""

            if code in overrides and abs_lap in overrides[code]:
                pit_this_lap = True
                pit_type     = "forced"
            elif (s.pits_done < max_stops
                  and _auto_pit_check(
                      s.tyre, age_this_lap, laps_left_after,
                      pit_loss, s.deg_rate,
                      _best_new_compound(s.tyre, s.pits_done))):
                pit_this_lap = True
                pit_type     = "auto"

            if pit_this_lap:
                new_compound = _best_new_compound(s.tyre, s.pits_done)
                lap_time    += pit_loss
                s.pit_laps.append({"lap": abs_lap, "compound": new_compound, "type": pit_type})
                s.tyre       = new_compound
                s.tyre_age   = 0
                s.pits_done += 1
                age_this_lap = 1
            else:
                s.tyre_age = age_this_lap

            s.cum_time += lap_time
            s.lap_times_sim.append(round(lap_time, 3))
            s.cum_series.append(round(s.cum_time, 3))

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Public simulator functions
# ─────────────────────────────────────────────────────────────────────────────

def simulate_race_to_finish(
    snapshot: RaceSnapshot,
    strategy_overrides: dict[str, list[int]] | None = None,
    max_stops: int = 2,
) -> RaceToFinishResult:
    """
    Project every driver from the current lap to the chequered flag.

    Each simulated lap applies:
      - Tyre degradation penalty (observed rate + cliff steepening)
      - Fuel burn gain (~0.035s/lap as weight reduces)
      - Automatic pit decision (stops when tyre cost > pit loss + fresh tyre cost)
      - Forced pit overrides via strategy_overrides

    Returns a RaceToFinishResult with final order, gap series, all pit events,
    and a key_numbers dict ready for LLM injection.

    Args:
        snapshot:           Current race state (from build_snapshot)
        strategy_overrides: {driver_code: [lap_numbers_to_pit]}
                            e.g. {"LEC": [45], "NOR": [38, 55]}
        max_stops:          Maximum pit stops per driver (default 2)
    """
    if not snapshot.drivers or snapshot.laps_remaining <= 0:
        return RaceToFinishResult(
            from_lap=snapshot.current_lap,
            total_laps=snapshot.total_laps,
            laps_simulated=0,
            confidence=0.0,
            assumptions=["Insufficient race data or no laps remaining"],
        )

    state = _run_simulation(snapshot, strategy_overrides, max_stops)

    # Build final order
    ordered = sorted(state.values(), key=lambda s: s.cum_time)
    winner_time = ordered[0].cum_time
    final_order = [
        {
            "position":      pos + 1,
            "driver_code":   s.code,
            "driver_name":   s.name,
            "gap_to_winner": round(s.cum_time - winner_time, 3),
            "pit_laps":      [p["lap"] for p in s.pit_laps],
            "pit_compounds": [p["compound"] for p in s.pit_laps],
            "final_tyre":    s.tyre,
            "final_tyre_age": s.tyre_age,
            "total_stops":   s.pits_done,
        }
        for pos, s in enumerate(ordered)
    ]

    # Gap series relative to leader (for charting)
    gap_series: dict[str, list[float]] = {}
    for code, s in state.items():
        leader_series = state[ordered[0].code].cum_series
        gap_series[code] = [
            round(s.cum_series[i] - leader_series[i], 3)
            for i in range(len(s.cum_series))
        ]

    # All pit events
    pit_events = []
    for s in state.values():
        for p in s.pit_laps:
            pit_events.append({
                "lap":      p["lap"],
                "driver":   s.name,
                "compound": p["compound"],
                "type":     p["type"],
            })
    pit_events.sort(key=lambda x: x["lap"])

    # Position changes vs current
    current_positions = {d.code: d.position for d in snapshot.drivers}
    position_changes = [
        {
            "driver": fo["driver_name"],
            "from_position": current_positions.get(fo["driver_code"], "?"),
            "to_position":   fo["position"],
            "delta": (current_positions.get(fo["driver_code"], fo["position"]) - fo["position"]),
        }
        for fo in final_order
        if current_positions.get(fo["driver_code"], fo["position"]) != fo["position"]
    ]
    position_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)

    key_numbers = {
        "race_name":        snapshot.race_name,
        "simulated_from_lap": snapshot.current_lap,
        "total_laps":       snapshot.total_laps,
        "laps_simulated":   snapshot.laps_remaining,
        "pit_loss_used_s":  snapshot.pit_loss_s,
        "fuel_gain_per_lap_s": _FUEL_GAIN_PER_LAP_S,
        "final_order_top5": final_order[:5],
        "position_changes": position_changes[:6],
        "pit_events":       pit_events,
        "strategy_overrides_applied": strategy_overrides or {},
        "winner_projected": final_order[0]["driver_name"] if final_order else "Unknown",
    }

    # Confidence degrades with laps remaining (more laps = more uncertainty)
    confidence = round(max(0.45, 0.90 - snapshot.laps_remaining * 0.008), 2)

    return RaceToFinishResult(
        from_lap=snapshot.current_lap,
        total_laps=snapshot.total_laps,
        laps_simulated=snapshot.laps_remaining,
        final_order=final_order,
        gap_series=gap_series,
        pit_events=pit_events,
        key_numbers=key_numbers,
        confidence=confidence,
        assumptions=[
            "Lap times projected using observed deg rate + cliff steepening",
            f"Fuel gain: {_FUEL_GAIN_PER_LAP_S}s/lap as fuel burns off",
            f"Pit loss: {snapshot.pit_loss_s}s ({snapshot.pit_stats.get('source','circuit default')})",
            "Auto-pit: fires when remaining tyre cost exceeds pit loss + new tyre cost",
            "No SC, red flag, or weather change modelled in this projection",
            "Driver errors, mechanical failures, and DRS battles not modelled",
        ],
    )


def simulate_strategy_comparison(
    snapshot: RaceSnapshot,
    strategies: dict[str, dict[str, list[int]]],
    max_stops: int = 2,
) -> ScenarioResult:
    """
    Compare two or more pit-stop strategies for one or more drivers.

    strategies: named dict of strategy_overrides dicts, e.g.
        {
            "1-stop (lap 45)":   {"LEC": [45]},
            "2-stop (lap 35+55)": {"LEC": [35, 55]},
            "stay out":           {},
        }

    For each named strategy, runs simulate_race_to_finish() and records
    the projected position and gap for each driver.

    Returns a ScenarioResult whose key_numbers contains:
        strategy_results:   {strategy_name: {driver: {position, gap, pit_laps}}}
        best_strategy:      name of the strategy with the best P1 gap for the
                            first driver mentioned across strategies
        delta_table:        pairwise time delta between every strategy pair
    """
    if not strategies:
        return ScenarioResult(
            scenario_type="strategy_comparison",
            trigger_lap=snapshot.current_lap,
            laps_remaining=snapshot.laps_remaining,
            affected_drivers=[],
            confidence=0.0,
            assumptions=["No strategies provided"],
        )

    strategy_results: dict[str, dict] = {}
    all_driver_codes: set[str] = set()

    for strat_name, overrides in strategies.items():
        result   = simulate_race_to_finish(snapshot, strategy_overrides=overrides,
                                           max_stops=max_stops)
        per_driver: dict[str, dict] = {}
        for fo in result.final_order:
            per_driver[fo["driver_code"]] = {
                "driver_name":   fo["driver_name"],
                "position":      fo["position"],
                "gap_to_winner": fo["gap_to_winner"],
                "pit_laps":      fo["pit_laps"],
                "pit_compounds": fo["pit_compounds"],
                "final_tyre":    fo["final_tyre"],
            }
            all_driver_codes.add(fo["driver_code"])
        strategy_results[strat_name] = per_driver

    # Find the primary driver (first code that appears across any override)
    primary_code: str | None = None
    for overrides in strategies.values():
        if overrides:
            primary_code = next(iter(overrides))
            break
    if primary_code is None and snapshot.drivers:
        primary_code = snapshot.drivers[0].code

    # Best strategy = lowest gap_to_winner for primary driver
    best_strategy: str | None = None
    best_gap = float("inf")
    for strat_name, per_driver in strategy_results.items():
        if primary_code and primary_code in per_driver:
            g = per_driver[primary_code]["gap_to_winner"]
            if g < best_gap:
                best_gap      = g
                best_strategy = strat_name

    # Pairwise delta table for the primary driver
    strat_names = list(strategy_results.keys())
    delta_table: dict[str, dict[str, float]] = {}
    if primary_code:
        for a in strat_names:
            delta_table[a] = {}
            for b in strat_names:
                ga = (strategy_results[a].get(primary_code) or {}).get("gap_to_winner", 0)
                gb = (strategy_results[b].get(primary_code) or {}).get("gap_to_winner", 0)
                delta_table[a][b] = round(ga - gb, 3)  # positive = A is worse than B

    primary_name = _driver_name(primary_code) if primary_code else "Unknown"

    return ScenarioResult(
        scenario_type="strategy_comparison",
        trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=list(all_driver_codes),
        key_numbers={
            "primary_driver":     primary_name,
            "strategies_compared": list(strategies.keys()),
            "strategy_results":   strategy_results,
            "best_strategy":      best_strategy,
            "best_strategy_gap_s": round(best_gap, 3),
            "delta_table":        delta_table,
            "race_name":          snapshot.race_name,
            "from_lap":           snapshot.current_lap,
            "laps_remaining":     snapshot.laps_remaining,
        },
        confidence=round(max(0.45, 0.88 - snapshot.laps_remaining * 0.008), 2),
        assumptions=[
            "Each strategy runs an independent lap-by-lap simulation",
            "All other drivers follow auto-pit decisions in each run",
            "No SC, weather, or mechanical changes modelled",
            f"Pit loss: {snapshot.pit_loss_s}s",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Existing simulators (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_safety_car(
    snapshot: RaceSnapshot,
    sc_type: str = "SC",
    sc_duration_laps: int = _SC_TYPICAL_DURATION_LAPS,
) -> ScenarioResult:
    compression = _SC_GAP_COMPRESSION_PER_LAP if sc_type == "SC" else _VSC_GAP_COMPRESSION_PER_LAP
    pit_loss_sc = snapshot.pit_loss_s * _SC_PIT_LOSS_FACTOR
    compressed  = {d.code: round(max(0.0, d.gap_to_leader - compression * sc_duration_laps), 2)
                   for d in snapshot.drivers}
    should_pit, analyses = [], {}
    for d in snapshot.drivers:
        if d.position > 10: continue
        params = _TYRE_DEG.get(d.tyre, _TYRE_DEG["U"])
        lpc    = max(0, d.tyre_age - params["cliff_lap"])
        rem_deg = d.tyre_deg_rate * snapshot.laps_remaining + lpc * 0.03 * snapshot.laps_remaining
        if d.throttle_mode == "PROTECTING" and d.throttle_drop_pct:
            rem_deg += d.tyre_deg_rate * 3
        net = rem_deg - pit_loss_sc
        analyses[d.code] = {"tyre": d.tyre, "tyre_age": d.tyre_age, "laps_past_cliff": lpc,
                             "throttle_mode": d.throttle_mode, "remaining_deg_s": round(rem_deg, 2),
                             "pit_loss_sc_s": round(pit_loss_sc, 2), "net_benefit_s": round(net, 2),
                             "verdict": "PIT" + (" (throttle urgent)" if d.throttle_mode == "PROTECTING" else "") if net > 0 else "STAY"}
        if net > 0: should_pit.append(d.code)
    eff  = {d.code: compressed[d.code] + (pit_loss_sc if d.code in should_pit else 0.0) for d in snapshot.drivers}
    srt  = sorted(eff.items(), key=lambda x: x[1])
    proj = {c: i+1 for i,(c,_) in enumerate(srt)}
    net_d= {d.code: round(eff[d.code] - d.gap_to_leader, 2) for d in snapshot.drivers}
    changes = sorted([{"driver":d.name,"from":d.position,"to":proj.get(d.code,d.position),
                        "delta":d.position-proj.get(d.code,d.position)}
                       for d in snapshot.drivers[:10] if d.position != proj.get(d.code,d.position)],
                      key=lambda x: abs(x["delta"]), reverse=True)
    return ScenarioResult(
        scenario_type=sc_type, trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=[d.code for d in snapshot.drivers[:10]],
        projected_positions=proj, net_time_deltas=net_d,
        key_numbers={"sc_type":sc_type,"sc_duration_laps":sc_duration_laps,
                     "gap_compression_total_s":round(compression*sc_duration_laps,1),
                     "pit_loss_normal_mean_s":round(snapshot.pit_stats.get("total_pit_loss_mean_s",snapshot.pit_loss_s),1),
                     "pit_loss_under_sc_s":round(pit_loss_sc,1),
                     "pit_data_source":snapshot.pit_stats.get("source","unknown"),
                     "drivers_should_pit":[_driver_name(c) for c in should_pit],
                     "position_changes":changes[:6],
                     "per_driver_analysis":{_driver_name(c):v for c,v in list(analyses.items())[:8]}},
        confidence=0.84,
        assumptions=[f"SC for ~{sc_duration_laps} laps",
                     f"Pit loss under {sc_type}: {round(pit_loss_sc,1)}s",
                     "Throttle protection mode increases pit urgency"])


def simulate_pit_stop(
    snapshot: RaceSnapshot,
    driver_code: str,
    pit_lap: int | None = None,
    new_compound: str | None = None,
) -> ScenarioResult:
    driver = next((d for d in snapshot.drivers if d.code.upper() == driver_code.upper()), None)
    if driver is None:
        return ScenarioResult(scenario_type="pit_stop", trigger_lap=snapshot.current_lap,
                              laps_remaining=snapshot.laps_remaining, affected_drivers=[driver_code],
                              confidence=0.0, assumptions=[f"Driver {driver_code} not found"])
    pit_lap        = pit_lap or snapshot.current_lap
    laps_after     = max(1, snapshot.total_laps - pit_lap)
    new_compound   = new_compound or ("M" if driver.tyre == "S" else ("H" if driver.tyre == "M" else "M"))
    tp             = _TYRE_DEG.get(driver.tyre, _TYRE_DEG["U"])
    ntp            = _TYRE_DEG.get(new_compound, _TYRE_DEG["M"])
    tc_mean        = snapshot.pit_stats.get("total_pit_loss_mean_s", snapshot.pit_loss_s)
    tc_best        = snapshot.pit_stats.get("total_pit_loss_p10_s",  snapshot.pit_loss_s - 0.5)
    tc_worst       = snapshot.pit_stats.get("total_pit_loss_p90_s",  snapshot.pit_loss_s + 1.2)
    lpc            = max(0, driver.tyre_age - tp["cliff_lap"])
    eff_deg        = driver.tyre_deg_rate + (driver.throttle_drop_pct or 0) * 0.04 \
                     if driver.throttle_mode == "PROTECTING" else driver.tyre_deg_rate
    cur_cost       = eff_deg * laps_after + lpc * 0.05 * laps_after
    new_cost       = ntp["deg_per_lap"] * laps_after
    pace_gain      = (ntp["pace_advantage_vs_m"] - tp.get("pace_advantage_vs_m", 0.0)) * laps_after * -1
    benefit        = cur_cost - new_cost + pace_gain
    net_mean       = round(tc_mean  - benefit, 2)
    net_best       = round(tc_best  - benefit, 2)
    net_worst      = round(tc_worst - benefit, 2)
    d_ahead        = next((d for d in snapshot.drivers if d.position == driver.position-1), None)
    d_behind       = next((d for d in snapshot.drivers if d.position == driver.position+1), None)
    gap_ahead      = round(driver.gap_to_leader - d_ahead.gap_to_leader,  2) if d_ahead  else None
    gap_behind     = round(d_behind.gap_to_leader - driver.gap_to_leader, 2) if d_behind else None
    rejoin_gap     = driver.gap_to_leader + tc_mean
    rejoin_pos     = sum(1 for d in snapshot.drivers if d.code != driver.code and d.gap_to_leader < rejoin_gap) + 1
    ltr            = round(abs(net_mean) / abs(ntp["pace_advantage_vs_m"] - tp.get("pace_advantage_vs_m",0.0)), 1) \
                     if net_mean < 0 and abs(ntp["pace_advantage_vs_m"] - tp.get("pace_advantage_vs_m",0.0)) > 0 else None
    ut             = {"from_driver":d_behind.name,"from_driver_tyre":f"{d_behind.tyre} age {d_behind.tyre_age}",
                      "current_gap_behind_s":gap_behind,
                      "threat_level":"HIGH" if gap_behind < 1.5 else "MEDIUM",
                      "note":"Could pit next lap and emerge ahead"} \
                     if d_behind and gap_behind is not None and gap_behind < 3.0 else None
    return ScenarioResult(
        scenario_type="pit_stop", trigger_lap=pit_lap, laps_remaining=laps_after,
        affected_drivers=[driver.code]+([d_behind.code] if d_behind else []),
        projected_positions={driver.code: rejoin_pos}, net_time_deltas={driver.code: net_mean},
        key_numbers={"driver":driver.name,"current_position":driver.position,
                     "pit_at_lap":pit_lap,"laps_remaining_after_pit":laps_after,
                     "current_tyre":driver.tyre,"current_tyre_age_laps":driver.tyre_age,
                     "laps_past_cliff":lpc,"throttle_mode":driver.throttle_mode,
                     "effective_deg_rate_s_per_lap":round(eff_deg,3),
                     "new_compound":new_compound,
                     "pit_loss_mean_s":round(tc_mean,2),"pit_loss_range":f"{round(tc_best,1)}–{round(tc_worst,1)}s",
                     "pit_data_source":snapshot.pit_stats.get("source","unknown"),
                     "tyre_benefit_s":round(benefit,2),"net_time_mean_s":net_mean,
                     "net_time_best_s":net_best,"net_time_worst_s":net_worst,
                     "rejoin_position":rejoin_pos,"gap_to_car_ahead_s":gap_ahead,
                     "gap_to_car_behind_s":gap_behind,"laps_to_recover_position":ltr,
                     "undercut_threat":ut,"car_ahead":d_ahead.name if d_ahead else None,
                     "car_behind":d_behind.name if d_behind else None,
                     "verdict":"PIT NOW — tyre benefit outweighs pit loss"
                               +(" (throttle protection confirms urgency)" if driver.throttle_mode=="PROTECTING" else "")
                               if net_mean < 0 else "HOLD — consider overcut or SC window"},
        confidence=0.82,
        assumptions=[f"Pit loss: {round(tc_mean,1)}s mean from {snapshot.pit_stats.get('source','circuit default')}",
                     f"New: {new_compound} deg {ntp['deg_per_lap']}s/lap",
                     f"Observed deg: {driver.tyre_deg_rate:.3f}s/lap"
                     +(f" → adjusted to {eff_deg:.3f} (throttle protecting)" if driver.throttle_mode=="PROTECTING" else "")])


def simulate_tyre_degradation(
    snapshot: RaceSnapshot,
    driver_codes: list[str] | None = None,
) -> ScenarioResult:
    codes = driver_codes or [d.code for d in snapshot.drivers[:6]]
    projections, warnings = {}, []
    for d in snapshot.drivers:
        if d.code not in codes: continue
        tp   = _TYRE_DEG.get(d.tyre, _TYRE_DEG["U"])
        ltc  = max(0, tp["cliff_lap"] - d.tyre_age)
        lpc  = max(0, d.tyre_age - tp["cliff_lap"])
        base = d.avg_lap_time_5 if d.avg_lap_time_5 > 60 else d.last_lap_time
        if base == 0: continue
        eff  = d.tyre_deg_rate + (d.throttle_drop_pct or 0)*0.04 if d.throttle_mode=="PROTECTING" else d.tyre_deg_rate
        proj = [round(base + eff*fl + max(0, d.tyre_age+fl-tp["cliff_lap"])*0.03, 3)
                for fl in range(1, min(snapshot.laps_remaining+1, 16))]
        tdl  = round(sum(proj) - base*len(proj), 2) if proj else 0.0
        tw   = d.throttle_mode == "PROTECTING"
        st   = "CRITICAL" if lpc>5 else "WARNING" if lpc>0 or ltc<=3 or tw else "OK"
        projections[d.name] = {"position":d.position,"tyre":d.tyre,"tyre_age_now":d.tyre_age,
                                "cliff_at_lap":tp["cliff_lap"],"laps_to_cliff":ltc,"laps_past_cliff":lpc,
                                "throttle_mode":d.throttle_mode,"throttle_drop_pct":d.throttle_drop_pct,
                                "observed_deg_s_per_lap":round(d.tyre_deg_rate,3),
                                "effective_deg_s_per_lap":round(eff,3),
                                "projected_lap_times_next_5":proj[:5],
                                "total_deg_loss_remaining_s":tdl,"status":st}
        if st in ("CRITICAL","WARNING"):
            warnings.append(f"{d.name} P{d.position} on {d.tyre} age {d.tyre_age} — {st}"
                            +(f" + throttle down {d.throttle_drop_pct:.1f}%" if tw else ""))
    return ScenarioResult(
        scenario_type="tyre_degradation", trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining, affected_drivers=codes,
        key_numbers={"driver_projections":projections,"cliff_warnings":warnings,
                     "laps_remaining":snapshot.laps_remaining},
        confidence=0.80,
        assumptions=["Deg from linear regression on same compound",
                     "Throttle protection adds 0.04s/lap per 1% drop",
                     "Cliff adds 0.03s/lap per lap beyond cliff"])


def simulate_overtake_window(
    snapshot: RaceSnapshot,
    attacker_code: str,
    defender_code: str,
) -> ScenarioResult:
    attacker = next((d for d in snapshot.drivers if d.code.upper()==attacker_code.upper()), None)
    defender = next((d for d in snapshot.drivers if d.code.upper()==defender_code.upper()), None)
    if not attacker or not defender:
        return ScenarioResult(scenario_type="overtake_window", trigger_lap=snapshot.current_lap,
                              laps_remaining=snapshot.laps_remaining,
                              affected_drivers=[attacker_code,defender_code], confidence=0.0)
    gap   = round(abs(attacker.gap_to_leader - defender.gap_to_leader), 3)
    att   = attacker.avg_lap_time_5 or attacker.last_lap_time
    dfn   = defender.avg_lap_time_5 or defender.last_lap_time
    pd    = round(att - dfn, 3)
    ltc   = round(gap/abs(pd),1) if pd<0 and gap>0 else None
    drs_e = gap <= _DRS_OVERTAKE_THRESHOLD_S
    easy  = gap <= _EASY_OVERTAKE_THRESHOLD_S
    ta    = round(defender.tyre_deg_rate*snapshot.laps_remaining - attacker.tyre_deg_rate*snapshot.laps_remaining, 2)
    cs    = snapshot.circuit_overtake.get("overtake_score", 0.5)
    cd    = snapshot.circuit_overtake.get("difficulty", "MODERATE")
    p     = round(0.05 + cs*0.25, 3)
    if pd < -0.3: p += 0.30
    if pd < -0.6: p += 0.10
    if drs_e:     p += 0.25
    if easy:      p += 0.15
    if ta > 2.0:  p += 0.20
    if attacker.gap_trend_3lap and attacker.gap_trend_3lap < -0.5: p += 0.10
    if attacker.tyre=="S" and defender.tyre=="H" and attacker.tyre_age<15: p += 0.10
    if defender.throttle_mode=="PROTECTING": p += 0.12
    if attacker.max_speed_delta and attacker.max_speed_delta > 3.0: p += 0.05
    p = min(0.95, round(p, 2))
    verdict = "HIGHLY LIKELY" if p>0.70 else "LIKELY" if p>0.50 else "POSSIBLE" if p>0.30 else "UNLIKELY"
    return ScenarioResult(
        scenario_type="overtake_window", trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining, affected_drivers=[attacker_code, defender_code],
        key_numbers={"attacker":attacker.name,"defender":defender.name,
                     "attacker_position":attacker.position,"defender_position":defender.position,
                     "current_gap_s":gap,"pace_delta_s_per_lap":pd,
                     "laps_to_close_at_current_pace":ltc,"drs_eligible":drs_e,
                     "circuit_difficulty":cd,"circuit_overtake_score":cs,
                     "circuit_drs_pct":snapshot.circuit_overtake.get("avg_drs_pct"),
                     "tyre_advantage_remaining_s":ta,
                     "attacker_tyre":f"{attacker.tyre} age {attacker.tyre_age}",
                     "defender_tyre":f"{defender.tyre} age {defender.tyre_age}",
                     "defender_throttle_mode":defender.throttle_mode,
                     "gap_trend_3lap_s":attacker.gap_trend_3lap,
                     "overtake_probability":p,"verdict":verdict},
        confidence=0.76,
        assumptions=[f"Circuit: {cd} (score {cs})",
                     "Pace delta from 5-lap rolling average",
                     "Defender protecting tyres adds +12% to attacker probability"])


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection — extended for Tier 2
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
    # Tier 2 intents
    "race_projection": [
        "who wins", "who will win", "final result", "end of race",
        "projected finish", "race outcome", "finish the race",
        "project the race", "simulate the race", "race to finish",
        "what happens if the race", "full race",
    ],
    "strategy_comparison": [
        "1-stop vs 2-stop", "one stop vs two stop", "compare strategy",
        "strategy comparison", "better strategy", "which strategy",
        "should he go 1 stop", "should she go 2 stop",
        "strategy options", "best strategy",
        "one stop or two stop", "1 stop vs 2 stop",
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
    q_upper, q_lower = query.upper(), query.lower()
    return [d.code for d in snapshot.drivers
            if d.code in q_upper or d.name.lower() in q_lower]


def _extract_lap_number(query: str) -> int | None:
    m = re.search(r'\blap\s+(\d+)\b', query, re.IGNORECASE)
    if m: return int(m.group(1))
    m = re.search(r'\bat\s+(\d+)\b', query, re.IGNORECASE)
    if m: return int(m.group(1))
    return None


def _extract_strategy_laps(query: str) -> list[int]:
    """Extract all lap numbers from queries like 'pit at lap 35 and lap 55'."""
    return [int(m) for m in re.findall(r'\blap\s+(\d+)\b', query, re.IGNORECASE)]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(
    query: str,
    session_id: str,
    live_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the appropriate scenario simulation for a natural language query.

    Scenarios (auto-detected from query):
        safety_car         SC/VSC gap compression + pit window
        pit_stop           Single driver pit analysis
        tyre_degradation   Lap time projection for top drivers
        overtake_window    Overtake probability assessment
        race_projection    Full lap-by-lap race to finish         ← NEW Tier 2
        strategy_comparison  Two-strategy comparison              ← NEW Tier 2
    """
    try:
        snapshot = build_snapshot(session_id, live_context)
        if snapshot is None or not snapshot.drivers:
            return {"snapshot": None, "scenario": None,
                    "scenario_type": "unknown",
                    "error": "No race data available — laps.json is empty or missing."}

        intent    = detect_scenario_intent(query)
        mentioned = _extract_driver_codes_from_query(query, snapshot)
        pit_lap   = _extract_lap_number(query)

        # ── Tier 2: race projection ───────────────────────────────────────────
        if intent == "race_projection":
            result_obj = simulate_race_to_finish(snapshot)
            scenario_dict = asdict(result_obj)
            snapshot_dict = _build_snapshot_dict(snapshot)
            return {"snapshot": snapshot_dict, "scenario": scenario_dict,
                    "scenario_type": intent, "error": None}

        # ── Tier 2: strategy comparison ───────────────────────────────────────
        if intent == "strategy_comparison":
            driver_code = mentioned[0] if mentioned else (
                snapshot.drivers[0].code if snapshot.drivers else "")
            laps_in_query = _extract_strategy_laps(query)

            # Build two default strategies when user hasn't specified laps
            if not laps_in_query:
                mid = snapshot.current_lap + max(5, snapshot.laps_remaining // 2)
                strategies = {
                    "stay out (no more stops)": {},
                    f"pit now (lap {snapshot.current_lap})": {driver_code: [snapshot.current_lap]},
                    f"pit at lap {mid}": {driver_code: [mid]},
                }
            elif len(laps_in_query) == 1:
                strategies = {
                    "stay out": {},
                    f"1-stop (lap {laps_in_query[0]})": {driver_code: [laps_in_query[0]]},
                }
            else:
                strategies = {
                    f"1-stop (lap {laps_in_query[0]})": {driver_code: [laps_in_query[0]]},
                    f"2-stop (lap {laps_in_query[0]}+{laps_in_query[1]})": {
                        driver_code: laps_in_query[:2]},
                }

            result = simulate_strategy_comparison(snapshot, strategies)
            snapshot_dict = _build_snapshot_dict(snapshot)
            return {"snapshot": snapshot_dict, "scenario": asdict(result),
                    "scenario_type": intent, "error": None}

        # ── Existing intents ──────────────────────────────────────────────────
        if intent == "safety_car":
            sc_type = "VSC" if any(k in query.lower() for k in ("vsc","virtual")) else "SC"
            result  = simulate_safety_car(snapshot, sc_type=sc_type)

        elif intent == "tyre_degradation":
            codes  = mentioned[:4] if mentioned else [d.code for d in snapshot.drivers[:6]]
            result = simulate_tyre_degradation(snapshot, driver_codes=codes)

        elif intent == "overtake_window":
            if len(mentioned) >= 2:
                attacker, defender = mentioned[0], mentioned[1]
            elif len(mentioned) == 1:
                d = next((dr for dr in snapshot.drivers if dr.code==mentioned[0]), None)
                if d and d.position > 1:
                    da = next((dr for dr in snapshot.drivers if dr.position==d.position-1), None)
                    attacker, defender = mentioned[0], da.code if da else snapshot.drivers[0].code
                else:
                    attacker = snapshot.drivers[1].code if len(snapshot.drivers)>1 else mentioned[0]
                    defender = snapshot.drivers[0].code
            else:
                attacker = snapshot.drivers[1].code if len(snapshot.drivers)>1 else ""
                defender = snapshot.drivers[0].code
            result = simulate_overtake_window(snapshot, attacker, defender)

        else:   # pit_stop
            driver_code = mentioned[0] if mentioned else next(
                (d.code for d in snapshot.drivers if d.position==1), snapshot.drivers[0].code)
            result = simulate_pit_stop(snapshot, driver_code=driver_code, pit_lap=pit_lap)

        snapshot_dict = _build_snapshot_dict(snapshot)
        return {"snapshot": snapshot_dict, "scenario": asdict(result),
                "scenario_type": intent, "error": None}

    except Exception as exc:
        logger.exception("Prediction engine error: %s", exc)
        return {"snapshot": None, "scenario": None,
                "scenario_type": "unknown", "error": str(exc)}


def _build_snapshot_dict(snapshot: RaceSnapshot) -> dict[str, Any]:
    return {
        "race_name":        snapshot.race_name,
        "track_name":       snapshot.track_name,
        "current_lap":      snapshot.current_lap,
        "total_laps":       snapshot.total_laps,
        "laps_remaining":   snapshot.laps_remaining,
        "pit_loss_s":       snapshot.pit_loss_s,
        "pit_stats":        snapshot.pit_stats,
        "circuit_overtake": snapshot.circuit_overtake,
        "top10_drivers": [
            {"pos": d.position, "code": d.code, "name": d.name,
             "gap_s": d.gap_to_leader, "gap_trend_3lap_s": d.gap_trend_3lap,
             "tyre": d.tyre, "tyre_age": d.tyre_age,
             "deg_rate": d.tyre_deg_rate, "avg_laptime_5": d.avg_lap_time_5,
             "pits_done": d.pit_stops_done, "throttle_mode": d.throttle_mode,
             "throttle_drop_pct": d.throttle_drop_pct,
             "max_speed_delta": d.max_speed_delta}
            for d in snapshot.drivers[:10]
        ],
    }
