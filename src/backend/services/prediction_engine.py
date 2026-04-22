"""
F1 Race Prediction Engine
=========================
Physics-grounded, data-driven scenario simulation.

Handles:
  - Safety Car / VSC gap compression + pit window analysis
  - Tyre degradation modelling with cliff detection
  - Pit stop undercut / overcut simulation
  - Leader pit scenario (rejoin position, net time)
  - DRS train / overtake probability by gap
  - Race outcome projection (laps remaining × pace delta)

All predictions are grounded in live laps.json data.
The LLM receives a pre-computed RaceSnapshot + ScenarioResult
so it reasons about real numbers, not generic templates.

Drop this file at: src/backend/services/prediction_engine.py
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.backend.config import settings
from src.backend.constants import DRIVER_CODES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# F1 Physics constants
# ─────────────────────────────────────────────────────────────────────────────

# Pit lane time-loss by circuit (seconds). Keyed on lowercase circuit fragments.
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

# Tyre degradation — lap time cost (seconds) per lap of tyre age.
# cliff_lap: age where pace penalty steepens sharply.
# deg_per_lap: seconds added per lap of age.
# pace_advantage_vs_m: raw pace delta vs medium (negative = faster).
_TYRE_DEG: dict[str, dict] = {
    "S": {"cliff_lap": 12, "deg_per_lap": 0.10, "pace_advantage_vs_m": -0.6},
    "M": {"cliff_lap": 25, "deg_per_lap": 0.06, "pace_advantage_vs_m":  0.0},
    "H": {"cliff_lap": 38, "deg_per_lap": 0.04, "pace_advantage_vs_m":  0.5},
    "I": {"cliff_lap": 20, "deg_per_lap": 0.07, "pace_advantage_vs_m":  0.3},
    "W": {"cliff_lap": 28, "deg_per_lap": 0.05, "pace_advantage_vs_m":  0.2},
    "U": {"cliff_lap": 20, "deg_per_lap": 0.07, "pace_advantage_vs_m":  0.0},
}

# Safety car / VSC parameters
_SC_GAP_COMPRESSION_PER_LAP  = 28.0   # s — field bunches per SC lap
_VSC_GAP_COMPRESSION_PER_LAP =  8.0   # s — VSC is less aggressive
_SC_PIT_LOSS_FACTOR           = 0.55   # pit loss is 55% of normal under SC
_SC_TYPICAL_DURATION_LAPS     = 4      # typical SC deployment in laps

# DRS / overtake thresholds
_DRS_OVERTAKE_THRESHOLD_S  = 1.0
_EASY_OVERTAKE_THRESHOLD_S = 0.5


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


def _pit_loss(track_name: str) -> float:
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
    last_lap_time: float           # seconds
    avg_lap_time_5: float          # 5-lap rolling average
    speed_kph: float
    pit_stops_done: int


@dataclass
class RaceSnapshot:
    """
    Everything the engine knows about the race right now.
    Built from laps.json + live_context. Passed to the LLM.
    """
    session_id: str
    race_name: str
    track_name: str
    current_lap: int
    total_laps: int
    laps_remaining: int
    pit_loss_s: float
    drivers: list[DriverState] = field(default_factory=list)
    recent_laps: list[dict] = field(default_factory=list)  # last 5 laps


@dataclass
class ScenarioResult:
    """
    Output of one scenario simulation.
    Contains the computed numbers — the LLM writes the narrative.
    """
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
# Lap data analytics
# ─────────────────────────────────────────────────────────────────────────────

def _observed_deg_rate(laps: list[dict], code: str, lookback: int = 5) -> float:
    """
    Estimate tyre deg rate for a driver from recent laps on the same compound.
    Returns seconds added per lap of tyre age. 0.0 if data is insufficient.
    Uses linear regression over the lookback window.
    """
    relevant: list[tuple[float, str]] = []
    for lap in reversed(laps[-lookback:]):
        lt = (lap.get("lap_times") or {}).get(code)
        tyre = (lap.get("tyres") or {}).get(code)
        if lt and lt > 60.0:
            relevant.append((float(lt), str(tyre or "U")))
        if len(relevant) >= lookback:
            break

    relevant.reverse()
    if len(relevant) < 3:
        return 0.0

    # Only use laps on the same compound (no pit stop in window)
    base_tyre = relevant[-1][1]
    same_stint = [lt for lt, t in relevant if t == base_tyre]
    if len(same_stint) < 2:
        return 0.0

    n = len(same_stint)
    x_mean = (n - 1) / 2
    y_mean = sum(same_stint) / n
    num = sum((i - x_mean) * (same_stint[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return round(num / den, 4)


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
# Snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshot(
    session_id: str,
    live_context: dict[str, Any],
    laps_override: list[dict] | None = None,
) -> RaceSnapshot | None:
    """
    Build a RaceSnapshot from laps.json + live_context.
    Returns None if insufficient data is available.
    """
    laps = laps_override if laps_override is not None else _load_laps(session_id)
    if not laps:
        return None

    meta = _load_metadata(session_id)
    track_name = meta.get("track_name") or live_context.get("race") or "Unknown"
    race_name = f"{meta.get('year', '')} {meta.get('gp', live_context.get('race', ''))}".strip()

    current_lap = int(live_context.get("current_lap") or laps[-1].get("lap", 0))
    total_laps = int(live_context.get("total_laps") or meta.get("total_laps", 0))
    laps_remaining = max(0, total_laps - current_lap)

    pit_loss = _pit_loss(track_name)
    latest_lap = laps[-1] if laps else {}

    positions_raw = live_context.get("positions") or latest_lap.get("positions") or {}
    gaps_raw = live_context.get("gaps") or latest_lap.get("gaps") or {}
    tyres_raw = live_context.get("tyres") or latest_lap.get("tyres") or {}
    tyre_ages_raw = live_context.get("tyre_ages") or latest_lap.get("tyre_ages") or {}
    avg_speed_raw = live_context.get("avg_speed") or latest_lap.get("avg_speed") or {}

    drivers: list[DriverState] = []
    for code, pos in sorted(positions_raw.items(), key=lambda x: x[1]):
        tyre = str(tyres_raw.get(code) or "U")
        tyre_age = int(tyre_ages_raw.get(code) or 1)

        deg_rate = _observed_deg_rate(laps, code)
        if deg_rate == 0.0:
            deg_rate = _TYRE_DEG.get(tyre, _TYRE_DEG["U"])["deg_per_lap"]

        last_lt = float((latest_lap.get("lap_times") or {}).get(code) or 0.0)
        avg_lt = _rolling_avg_laptime(laps, code, n=5)

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
        ))

    return RaceSnapshot(
        session_id=session_id,
        race_name=race_name,
        track_name=track_name,
        current_lap=current_lap,
        total_laps=total_laps,
        laps_remaining=laps_remaining,
        pit_loss_s=pit_loss,
        drivers=drivers,
        recent_laps=laps[-5:],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario simulators
# ─────────────────────────────────────────────────────────────────────────────

def simulate_safety_car(
    snapshot: RaceSnapshot,
    sc_type: str = "SC",
    sc_duration_laps: int = _SC_TYPICAL_DURATION_LAPS,
) -> ScenarioResult:
    """
    SC / VSC gap compression, pit window evaluation, projected restart order.
    """
    compression_per_lap = (
        _SC_GAP_COMPRESSION_PER_LAP if sc_type == "SC"
        else _VSC_GAP_COMPRESSION_PER_LAP
    )
    pit_loss_sc = snapshot.pit_loss_s * _SC_PIT_LOSS_FACTOR

    # How much each gap compresses
    compressed_gaps: dict[str, float] = {}
    for d in snapshot.drivers:
        compressed = max(0.0, d.gap_to_leader - compression_per_lap * sc_duration_laps)
        compressed_gaps[d.code] = round(compressed, 2)

    # For each driver: should they pit?
    should_pit: list[str] = []
    pit_analyses: dict[str, dict] = {}

    for d in snapshot.drivers:
        if d.position > 10:
            continue
        tyre_params = _TYRE_DEG.get(d.tyre, _TYRE_DEG["U"])
        laps_past_cliff = max(0, d.tyre_age - tyre_params["cliff_lap"])
        remaining_deg_loss = (
            d.tyre_deg_rate * snapshot.laps_remaining
            + laps_past_cliff * 0.03 * snapshot.laps_remaining
        )
        net_benefit = remaining_deg_loss - pit_loss_sc

        pit_analyses[d.code] = {
            "tyre_age": d.tyre_age,
            "tyre": d.tyre,
            "laps_past_cliff": laps_past_cliff,
            "remaining_deg_loss_s": round(remaining_deg_loss, 2),
            "pit_loss_under_sc_s": round(pit_loss_sc, 2),
            "net_pit_benefit_s": round(net_benefit, 2),
            "verdict": "PIT" if net_benefit > 0 else "STAY",
        }
        if net_benefit > 0:
            should_pit.append(d.code)

    # Projected effective race time after SC + optional pit
    effective_times: dict[str, float] = {}
    for d in snapshot.drivers:
        t = compressed_gaps[d.code]
        if d.code in should_pit:
            t += pit_loss_sc
        effective_times[d.code] = t

    sorted_after = sorted(effective_times.items(), key=lambda x: x[1])
    projected_positions = {code: i + 1 for i, (code, _) in enumerate(sorted_after)}
    net_deltas = {
        d.code: round(effective_times[d.code] - d.gap_to_leader, 2)
        for d in snapshot.drivers
    }

    # Position changes
    pos_changes = []
    for d in snapshot.drivers[:10]:
        new_pos = projected_positions.get(d.code, d.position)
        delta = d.position - new_pos  # positive = gained positions
        if delta != 0:
            pos_changes.append({
                "driver": d.name,
                "from": d.position,
                "to": new_pos,
                "delta": delta,
            })
    pos_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return ScenarioResult(
        scenario_type=sc_type,
        trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=[d.code for d in snapshot.drivers[:10]],
        projected_positions=projected_positions,
        net_time_deltas=net_deltas,
        key_numbers={
            "sc_type": sc_type,
            "sc_duration_laps": sc_duration_laps,
            "pit_loss_normal_s": round(snapshot.pit_loss_s, 1),
            "pit_loss_under_sc_s": round(pit_loss_sc, 1),
            "gap_compression_total_s": round(compression_per_lap * sc_duration_laps, 1),
            "drivers_should_pit": [_driver_name(c) for c in should_pit],
            "position_changes": pos_changes[:6],
            "per_driver_analysis": {
                _driver_name(c): v
                for c, v in list(pit_analyses.items())[:8]
            },
        },
        confidence=0.82,
        assumptions=[
            f"SC deployed for ~{sc_duration_laps} laps",
            f"Pit loss under {sc_type}: {round(pit_loss_sc, 1)}s (vs normal {round(snapshot.pit_loss_s, 1)}s)",
            "Gap compression modelled as linear per SC lap",
            "Tyre degradation from observed pace trend + compound model",
            "Restart order: gap after SC + any pit-stop time added",
        ],
    )


def simulate_pit_stop(
    snapshot: RaceSnapshot,
    driver_code: str,
    pit_lap: int | None = None,
    new_compound: str | None = None,
) -> ScenarioResult:
    """
    Simulate a pit stop for a specific driver.
    Computes: net time cost, rejoin position, undercut risk, and whether it's worth it.
    """
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

    pit_lap = pit_lap or snapshot.current_lap
    laps_after_pit = max(1, snapshot.total_laps - pit_lap)

    # Select fresh compound
    if new_compound is None:
        new_compound = "M" if driver.tyre == "S" else ("H" if driver.tyre == "M" else "M")

    tyre_params = _TYRE_DEG.get(driver.tyre, _TYRE_DEG["U"])
    new_tyre_params = _TYRE_DEG.get(new_compound, _TYRE_DEG["M"])

    # Time cost in pit
    time_cost = snapshot.pit_loss_s

    # Remaining deg loss on current tyre
    laps_past_cliff = max(0, driver.tyre_age - tyre_params["cliff_lap"])
    current_deg = driver.tyre_deg_rate * laps_after_pit + laps_past_cliff * 0.05 * laps_after_pit

    # Expected deg on fresh tyre
    new_deg = new_tyre_params["deg_per_lap"] * laps_after_pit

    # Pace delta between compounds × remaining laps
    pace_delta_per_lap = new_tyre_params["pace_advantage_vs_m"] - tyre_params.get("pace_advantage_vs_m", 0.0)
    pace_gain = pace_delta_per_lap * laps_after_pit * -1  # negative = faster on new

    tyre_time_benefit = current_deg - new_deg + pace_gain
    net_time = round(time_cost - tyre_time_benefit, 2)

    # Cars around the driver
    d_ahead = next((d for d in snapshot.drivers if d.position == driver.position - 1), None)
    d_behind = next((d for d in snapshot.drivers if d.position == driver.position + 1), None)

    gap_ahead = round(driver.gap_to_leader - d_ahead.gap_to_leader, 2) if d_ahead else None
    gap_behind = round(d_behind.gap_to_leader - driver.gap_to_leader, 2) if d_behind else None

    # Rejoin position — after pit the driver is effectively pit_loss behind
    rejoin_gap = driver.gap_to_leader + time_cost
    rejoin_pos = sum(1 for d in snapshot.drivers if d.code != driver.code and d.gap_to_leader < rejoin_gap) + 1

    # Final position estimate: if net_time is negative (net gain), may recover
    laps_to_recover = None
    if net_time < 0 and d_behind:
        # How many laps to make up the rejoin gap at observed pace advantage
        pace_adv = abs(pace_delta_per_lap)
        if pace_adv > 0:
            laps_to_recover = round((time_cost - tyre_time_benefit) / pace_adv, 1)

    undercut_threat = None
    if d_behind and gap_behind is not None and gap_behind < 3.0:
        undercut_threat = {
            "from_driver": d_behind.name,
            "current_gap_behind_s": gap_behind,
            "threat_level": "HIGH" if gap_behind < 1.5 else "MEDIUM",
            "note": "They could pit next lap and emerge ahead on fresh rubber",
        }

    return ScenarioResult(
        scenario_type="pit_stop",
        trigger_lap=pit_lap,
        laps_remaining=laps_after_pit,
        affected_drivers=[driver.code] + ([d_behind.code] if d_behind else []),
        projected_positions={driver.code: rejoin_pos},
        net_time_deltas={driver.code: net_time},
        key_numbers={
            "driver": driver.name,
            "current_position": driver.position,
            "pit_at_lap": pit_lap,
            "laps_remaining_after_pit": laps_after_pit,
            "current_tyre": driver.tyre,
            "current_tyre_age_laps": driver.tyre_age,
            "laps_past_cliff": laps_past_cliff,
            "new_compound": new_compound,
            "pit_loss_s": round(snapshot.pit_loss_s, 1),
            "tyre_time_benefit_s": round(tyre_time_benefit, 2),
            "net_time_cost_s": net_time,
            "rejoin_position": rejoin_pos,
            "gap_to_car_ahead_s": gap_ahead,
            "gap_to_car_behind_s": gap_behind,
            "laps_to_recover_position": laps_to_recover,
            "undercut_threat": undercut_threat,
            "car_ahead": d_ahead.name if d_ahead else None,
            "car_behind": d_behind.name if d_behind else None,
            "verdict": (
                "PIT NOW — tyre benefit outweighs pit loss"
                if net_time < 0
                else "HOLD — pit loss exceeds projected tyre gain; consider overcut or wait for SC"
            ),
        },
        confidence=0.78,
        assumptions=[
            f"Pit loss at {snapshot.track_name}: {round(snapshot.pit_loss_s, 1)}s",
            f"New compound: {new_compound}, nominal deg: {new_tyre_params['deg_per_lap']}s/lap",
            f"Current observed deg rate: {driver.tyre_deg_rate:.3f}s/lap",
            "Rejoin assumed clear unless gap_to_behind < 3s",
        ],
    )


def simulate_tyre_degradation(
    snapshot: RaceSnapshot,
    driver_codes: list[str] | None = None,
) -> ScenarioResult:
    """
    Project lap time evolution for key drivers over remaining laps.
    Identifies who is at or past the degradation cliff.
    """
    codes = driver_codes or [d.code for d in snapshot.drivers[:6]]
    projections: dict[str, dict] = {}
    cliff_warnings: list[str] = []

    for d in snapshot.drivers:
        if d.code not in codes:
            continue
        tyre_params = _TYRE_DEG.get(d.tyre, _TYRE_DEG["U"])
        laps_to_cliff = max(0, tyre_params["cliff_lap"] - d.tyre_age)
        laps_past_cliff = max(0, d.tyre_age - tyre_params["cliff_lap"])

        base_time = d.avg_lap_time_5 if d.avg_lap_time_5 > 60 else d.last_lap_time
        if base_time == 0:
            continue

        # Project pace for each future lap
        lap_projections = []
        for future_lap in range(1, min(snapshot.laps_remaining + 1, 16)):
            future_age = d.tyre_age + future_lap
            deg_penalty = d.tyre_deg_rate * future_lap
            cliff_extra = max(0, future_age - tyre_params["cliff_lap"]) * 0.03
            projected_time = base_time + deg_penalty + cliff_extra
            lap_projections.append(round(projected_time, 3))

        total_deg_loss = round(
            sum(lap_projections) - base_time * len(lap_projections), 2
        ) if lap_projections else 0.0

        status = (
            "CRITICAL" if laps_past_cliff > 5 else
            "WARNING" if laps_past_cliff > 0 or laps_to_cliff <= 3 else
            "OK"
        )

        projections[d.name] = {
            "position": d.position,
            "tyre": d.tyre,
            "tyre_age_now": d.tyre_age,
            "cliff_at_lap": tyre_params["cliff_lap"],
            "laps_to_cliff": laps_to_cliff,
            "laps_past_cliff": laps_past_cliff,
            "deg_rate_s_per_lap": round(d.tyre_deg_rate, 3),
            "projected_lap_times_next_5": lap_projections[:5],
            "total_deg_loss_remaining_s": total_deg_loss,
            "status": status,
        }

        if status in ("CRITICAL", "WARNING"):
            cliff_warnings.append(f"{d.name} P{d.position} on {d.tyre} age {d.tyre_age} — {status}")

    return ScenarioResult(
        scenario_type="tyre_degradation",
        trigger_lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        affected_drivers=codes,
        key_numbers={
            "driver_projections": projections,
            "cliff_warnings": cliff_warnings,
            "laps_remaining": snapshot.laps_remaining,
        },
        confidence=0.75,
        assumptions=[
            "Deg rate from linear regression over last 5 laps on same compound",
            "Cliff steepness adds 0.03s/lap per lap beyond the cliff",
            "Weather assumed stable (no wet/dry crossover)",
        ],
    )


def simulate_overtake_window(
    snapshot: RaceSnapshot,
    attacker_code: str,
    defender_code: str,
) -> ScenarioResult:
    """
    Assess whether an overtake is likely: pace delta, DRS window, tyre state.
    """
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

    att_lt = attacker.avg_lap_time_5 or attacker.last_lap_time
    def_lt = defender.avg_lap_time_5 or defender.last_lap_time
    pace_delta = round(att_lt - def_lt, 3)  # negative = attacker faster

    laps_to_close = None
    if pace_delta < 0 and current_gap > 0:
        laps_to_close = round(current_gap / abs(pace_delta), 1)

    drs_eligible = current_gap <= _DRS_OVERTAKE_THRESHOLD_S
    easy_drs = current_gap <= _EASY_OVERTAKE_THRESHOLD_S

    # Tyre advantage over remaining laps
    att_remaining_deg = attacker.tyre_deg_rate * snapshot.laps_remaining
    def_remaining_deg = defender.tyre_deg_rate * snapshot.laps_remaining
    tyre_advantage = round(def_remaining_deg - att_remaining_deg, 2)

    # Probability model
    p = 0.10
    if pace_delta < -0.3:
        p += 0.30
    if pace_delta < -0.6:
        p += 0.10
    if drs_eligible:
        p += 0.25
    if easy_drs:
        p += 0.15
    if tyre_advantage > 2.0:
        p += 0.20
    if attacker.gap_trend_3lap and attacker.gap_trend_3lap < -0.5:
        p += 0.10
    if attacker.tyre in ("S",) and defender.tyre in ("H",) and attacker.tyre_age < 15:
        p += 0.10  # fresh soft vs worn hard
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
            "attacker": attacker.name,
            "defender": defender.name,
            "attacker_position": attacker.position,
            "defender_position": defender.position,
            "current_gap_s": current_gap,
            "pace_delta_s_per_lap": pace_delta,
            "laps_to_close_at_current_pace": laps_to_close,
            "drs_eligible": drs_eligible,
            "tyre_advantage_over_remaining_s": tyre_advantage,
            "attacker_tyre": f"{attacker.tyre} age {attacker.tyre_age}",
            "defender_tyre": f"{defender.tyre} age {defender.tyre_age}",
            "gap_trend_3lap_s": attacker.gap_trend_3lap,
            "overtake_probability": p,
            "verdict": verdict,
        },
        confidence=0.72,
        assumptions=[
            "DRS effectiveness: ~0.3–0.5s advantage per lap",
            "Pace delta from 5-lap rolling average",
            "Track position advantage assumed to defender",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
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
    """Return the most likely scenario type from a query string."""
    q = query.lower()
    scores: dict[str, int] = {k: 0 for k in _SCENARIO_KEYWORDS}
    for scenario, keywords in _SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[scenario] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "pit_stop"


def _extract_driver_codes_from_query(query: str, snapshot: RaceSnapshot) -> list[str]:
    """Extract mentioned driver codes or names from a query."""
    q_upper = query.upper()
    q_lower = query.lower()
    found = []
    for d in snapshot.drivers:
        if d.code in q_upper or d.name.lower() in q_lower:
            found.append(d.code)
    return found


def _extract_lap_number(query: str) -> int | None:
    """Extract a specific lap number from 'lap 35' or 'at lap 40' etc."""
    m = re.search(r'\blap\s+(\d+)\b', query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'\bat\s+(\d+)\b', query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


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

    Returns a dict with:
      snapshot  — serialised RaceSnapshot (top-10 driver states)
      scenario  — serialised ScenarioResult (computed numbers)
      scenario_type — str
      error     — str | None
    """
    try:
        snapshot = build_snapshot(session_id, live_context)
        if snapshot is None or not snapshot.drivers:
            return {
                "snapshot": None,
                "scenario": None,
                "scenario_type": "unknown",
                "error": "No race data available — laps.json is empty or missing.",
            }

        intent = detect_scenario_intent(query)
        mentioned = _extract_driver_codes_from_query(query, snapshot)
        pit_lap = _extract_lap_number(query)

        if intent == "safety_car":
            sc_type = "VSC" if any(k in query.lower() for k in ("vsc", "virtual")) else "SC"
            result = simulate_safety_car(snapshot, sc_type=sc_type)

        elif intent == "tyre_degradation":
            codes = mentioned[:4] if mentioned else [d.code for d in snapshot.drivers[:6]]
            result = simulate_tyre_degradation(snapshot, driver_codes=codes)

        elif intent == "overtake_window":
            if len(mentioned) >= 2:
                attacker, defender = mentioned[0], mentioned[1]
            elif len(mentioned) == 1:
                d = next((dr for dr in snapshot.drivers if dr.code == mentioned[0]), None)
                if d and d.position > 1:
                    d_ahead = next((dr for dr in snapshot.drivers if dr.position == d.position - 1), None)
                    attacker = mentioned[0]
                    defender = d_ahead.code if d_ahead else snapshot.drivers[0].code
                else:
                    attacker = snapshot.drivers[1].code if len(snapshot.drivers) > 1 else mentioned[0]
                    defender = snapshot.drivers[0].code
            else:
                attacker = snapshot.drivers[1].code if len(snapshot.drivers) > 1 else ""
                defender = snapshot.drivers[0].code
            result = simulate_overtake_window(snapshot, attacker, defender)

        else:  # pit_stop (default)
            driver_code = mentioned[0] if mentioned else next(
                (d.code for d in snapshot.drivers if d.position == 1), snapshot.drivers[0].code
            )
            result = simulate_pit_stop(snapshot, driver_code=driver_code, pit_lap=pit_lap)

        snapshot_dict = {
            "race_name": snapshot.race_name,
            "track_name": snapshot.track_name,
            "current_lap": snapshot.current_lap,
            "total_laps": snapshot.total_laps,
            "laps_remaining": snapshot.laps_remaining,
            "pit_loss_s": snapshot.pit_loss_s,
            "top10_drivers": [
                {
                    "pos": d.position,
                    "code": d.code,
                    "name": d.name,
                    "gap_s": d.gap_to_leader,
                    "gap_trend_3lap_s": d.gap_trend_3lap,
                    "tyre": d.tyre,
                    "tyre_age": d.tyre_age,
                    "deg_rate_s_per_lap": d.tyre_deg_rate,
                    "avg_laptime_5": d.avg_lap_time_5,
                    "pits_done": d.pit_stops_done,
                }
                for d in snapshot.drivers[:10]
            ],
        }

        return {
            "snapshot": snapshot_dict,
            "scenario": asdict(result),
            "scenario_type": intent,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Prediction engine error: %s", exc)
        return {
            "snapshot": None,
            "scenario": None,
            "scenario_type": "unknown",
            "error": str(exc),
        }
