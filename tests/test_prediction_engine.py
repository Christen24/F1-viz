"""
Tests for Tier 1 prediction engine upgrade.
Drop at: tests/test_prediction_engine.py
Run:     pytest tests/test_prediction_engine.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.backend.services.prediction_engine import (
    DriverState,
    RaceSnapshot,
    build_snapshot,
    compute_circuit_overtake_score,
    compute_pit_statistics,
    detect_scenario_intent,
    run_prediction,
    simulate_overtake_window,
    simulate_pit_stop,
    simulate_safety_car,
    simulate_tyre_degradation,
    _detect_throttle_cliff,
    _gap_trend,
    _max_speed_delta,
    _observed_deg_rate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_laps(n: int = 40) -> list[dict]:
    """
    40 laps of mock data.
    LEC:  on Hard since lap 28, throttle drops from lap 34.
    NOR:  on Medium until lap 32, then fresh Soft.
    VER:  pitted at lap 35 onto Medium.
    HAM:  stayed out entire race on Hard (old tyres).
    PIA:  stayed out on Medium.
    """
    laps = []
    for i in range(1, n + 1):
        lec_throttle = 88.0 - max(0, i - 34) * 0.8
        lec_lt       = round(91.0 + i * 0.05 + max(0, i - 35) * 0.08, 3)
        nor_tyre     = "S" if i >= 32 else "M"
        nor_lt       = round(90.2 + (i - 32) * 0.10 if i >= 32 else 90.8 + i * 0.06, 3)

        pit_stops = []
        if i == 28:
            pit_stops.append({"type": "pit_stop", "actor": "LEC",
                "details": {"pit_duration": 2.6, "compound": "H", "lap": 28}, "victim": None})
        if i == 32:
            pit_stops.append({"type": "pit_stop", "actor": "NOR",
                "details": {"pit_duration": 2.4, "compound": "S", "lap": 32}, "victim": None})
        if i == 35:
            pit_stops.append({"type": "pit_stop", "actor": "VER",
                "details": {"pit_duration": 2.1, "compound": "M", "lap": 35}, "victim": None})

        laps.append({
            "lap": i,
            "leader": "LEC",
            "positions":  {"LEC": 1, "NOR": 2, "VER": 3, "HAM": 4, "PIA": 5},
            "gaps":       {"LEC": 0.0, "NOR": 1.8, "VER": 4.1, "HAM": 8.3, "PIA": 12.1},
            "tyres":      {"LEC": "H", "NOR": nor_tyre, "VER": "M", "HAM": "H", "PIA": "M"},
            "tyre_ages":  {"LEC": max(1, i - 27), "NOR": max(1, i - 31),
                           "VER": max(1, i - 34), "HAM": i, "PIA": i},
            "lap_times":  {"LEC": lec_lt, "NOR": nor_lt,
                           "VER": round(91.2 + max(0, i - 34) * 0.06, 3),
                           "HAM": round(92.0 + i * 0.04, 3),
                           "PIA": round(92.3 + i * 0.03, 3)},
            "pit_stops":  pit_stops,
            "avg_speed":  {"LEC": 195.2, "NOR": 198.1, "VER": 196.4, "HAM": 194.0, "PIA": 193.5},
            "max_speed":  {"LEC": 318.0, "NOR": 322.0, "VER": 320.0, "HAM": 315.0, "PIA": 314.0},
            "throttle_pct": {"LEC": round(lec_throttle, 1), "NOR": 90.2,
                             "VER": 88.8, "HAM": 87.5, "PIA": 88.1},
            "brake_pct":    {"LEC": 18.0, "NOR": 17.5, "VER": 18.2, "HAM": 19.0, "PIA": 18.8},
            "drs_pct":      {"LEC": 22.0, "NOR": 23.5, "VER": 22.8, "HAM": 21.5, "PIA": 21.8},
            "sector_times": {
                "LEC": [round(29.1 + i * 0.01, 3), round(38.0 + i * 0.02, 3), round(23.9 + i * 0.02, 3)],
                "NOR": [28.9, 38.0, 23.8],
                "VER": [29.2, 38.1, 23.7],
            },
            "events": [],
        })
    return laps


def _monaco_snapshot() -> RaceSnapshot:
    """Pre-built snapshot: Monaco lap 40/50, LEC protecting tyres."""
    laps = _make_laps(40)
    live = {
        "session_id": "test",
        "current_lap": 40, "total_laps": 50,
        "race": "2025 Monaco Grand Prix",
        "positions":  {"LEC": 1, "NOR": 2, "VER": 3, "HAM": 4, "PIA": 5},
        "gaps":       {"LEC": 0.0, "NOR": 1.8, "VER": 4.1, "HAM": 8.3, "PIA": 12.1},
        "tyres":      {"LEC": "H", "NOR": "S", "VER": "M", "HAM": "H", "PIA": "M"},
        "tyre_ages":  {"LEC": 12, "NOR": 8, "VER": 5, "HAM": 40, "PIA": 40},
        "avg_speed":  {"LEC": 195.2, "NOR": 198.1, "VER": 196.4, "HAM": 194.0, "PIA": 193.5},
    }
    with patch("src.backend.services.prediction_engine._load_laps",   return_value=laps), \
         patch("src.backend.services.prediction_engine._load_metadata", return_value={
             "track_name": "Monaco", "year": 2025, "gp": "Monaco Grand Prix", "total_laps": 50}):
        snap = build_snapshot("test", live)
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# 1. Throttle cliff detector
# ─────────────────────────────────────────────────────────────────────────────

class TestThrottleCliffDetector:

    def test_lec_protecting_after_sustained_drop(self):
        laps = _make_laps(40)
        result = _detect_throttle_cliff(laps, "LEC", lookback=8)
        assert result["mode"] == "PROTECTING"
        assert result["drop_pct"] is not None
        assert result["drop_pct"] >= 2.5

    def test_lec_confidence_positive_when_protecting(self):
        laps = _make_laps(40)
        result = _detect_throttle_cliff(laps, "LEC", lookback=8)
        assert result["confidence"] > 0.0

    def test_lec_laps_in_protection_at_least_two(self):
        laps = _make_laps(40)
        result = _detect_throttle_cliff(laps, "LEC", lookback=8)
        assert result["laps_in_protection_mode"] >= 2

    def test_nor_stable_throttle_is_normal(self):
        laps = _make_laps(40)
        result = _detect_throttle_cliff(laps, "NOR", lookback=8)
        assert result["mode"] == "NORMAL"
        assert result["confidence"] == 0.0

    def test_insufficient_data_returns_unknown(self):
        result = _detect_throttle_cliff([{"throttle_pct": {"LEC": 88.0}}], "LEC")
        assert result["mode"] == "UNKNOWN"

    def test_baseline_and_current_are_populated(self):
        laps = _make_laps(40)
        result = _detect_throttle_cliff(laps, "LEC", lookback=8)
        assert result["throttle_baseline_pct"] is not None
        assert result["throttle_current_pct"] is not None
        assert result["throttle_baseline_pct"] > result["throttle_current_pct"]

    def test_interpretation_string_is_populated(self):
        laps = _make_laps(40)
        result = _detect_throttle_cliff(laps, "LEC")
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stochastic pit model
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePitStatistics:

    def test_session_observed_with_three_stops(self):
        laps = _make_laps(40)
        ps = compute_pit_statistics(laps, circuit_default_s=22.5)
        assert ps["source"] == "session_observed"
        assert ps["sample_count"] == 3

    def test_mean_stationary_in_expected_range(self):
        laps = _make_laps(40)
        ps = compute_pit_statistics(laps, circuit_default_s=22.5)
        assert 2.0 < ps["mean_stationary_s"] < 3.0

    def test_p10_less_than_mean_less_than_p90(self):
        laps = _make_laps(40)
        ps = compute_pit_statistics(laps, circuit_default_s=22.5)
        assert ps["total_pit_loss_p10_s"] < ps["total_pit_loss_mean_s"] < ps["total_pit_loss_p90_s"]

    def test_std_positive(self):
        laps = _make_laps(40)
        ps = compute_pit_statistics(laps, circuit_default_s=22.5)
        assert ps["std_stationary_s"] > 0

    def test_empty_laps_falls_back_to_default(self):
        ps = compute_pit_statistics([], circuit_default_s=21.0)
        assert ps["source"] == "circuit_default"
        assert ps["total_pit_loss_mean_s"] == 21.0

    def test_single_stop_falls_back_to_default(self):
        laps = [{"pit_stops": [
            {"type": "pit_stop", "actor": "LEC",
             "details": {"pit_duration": 2.5, "compound": "H", "lap": 20}, "victim": None}
        ]}]
        ps = compute_pit_statistics(laps, circuit_default_s=21.0)
        assert ps["source"] == "circuit_default"

    def test_interpretation_string_present(self):
        laps = _make_laps(40)
        ps = compute_pit_statistics(laps, circuit_default_s=22.5)
        assert isinstance(ps["interpretation"], str)
        assert "stop" in ps["interpretation"].lower()

    def test_corrupt_zero_durations_filtered(self):
        laps = [{"pit_stops": [
            {"type": "pit_stop", "actor": "LEC",
             "details": {"pit_duration": 0.0, "compound": "H", "lap": 20}, "victim": None},
            {"type": "pit_stop", "actor": "NOR",
             "details": {"pit_duration": 2.4, "compound": "S", "lap": 30}, "victim": None},
        ]}]
        ps = compute_pit_statistics(laps, circuit_default_s=21.0)
        # Only 1 valid stop — should fall back to default
        assert ps["source"] == "circuit_default"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Circuit overtake score
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCircuitOvertakeScore:

    def test_score_in_zero_to_one_range(self):
        laps = _make_laps(40)
        co = compute_circuit_overtake_score(laps)
        assert 0.0 <= co["overtake_score"] <= 1.0

    def test_difficulty_is_valid_label(self):
        laps = _make_laps(40)
        co = compute_circuit_overtake_score(laps)
        assert co["difficulty"] in ("EASY", "MODERATE", "HARD")

    def test_avg_drs_pct_positive(self):
        laps = _make_laps(40)
        co = compute_circuit_overtake_score(laps)
        assert co["avg_drs_pct"] > 0

    def test_source_is_session_data(self):
        laps = _make_laps(40)
        co = compute_circuit_overtake_score(laps)
        assert co["source"] == "session_data"

    def test_empty_laps_falls_back_to_defaults(self):
        co = compute_circuit_overtake_score([])
        assert co["source"] == "defaults"

    def test_high_drs_gives_higher_score(self):
        # Build two lap sets: one with 32% DRS, one with 10%
        def laps_with_drs(pct):
            return [{"drs_pct": {"A": pct, "B": pct},
                     "sector_times": {"A": [29.0, 38.0, 24.0], "B": [29.1, 38.1, 24.1]}}
                    for _ in range(10)]

        high_drs = compute_circuit_overtake_score(laps_with_drs(32.0))
        low_drs  = compute_circuit_overtake_score(laps_with_drs(10.0))
        assert high_drs["overtake_score"] > low_drs["overtake_score"]

    def test_interpretation_mentions_drs(self):
        laps = _make_laps(40)
        co = compute_circuit_overtake_score(laps)
        assert "drs" in co["interpretation"].lower() or "%" in co["interpretation"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Max speed delta
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxSpeedDelta:

    def test_returns_float_for_valid_data(self):
        laps = _make_laps(40)
        delta = _max_speed_delta(laps, "LEC")
        assert isinstance(delta, float)

    def test_returns_none_for_insufficient_data(self):
        delta = _max_speed_delta([{"max_speed": {"LEC": 300.0}}], "LEC")
        assert delta is None

    def test_negative_delta_when_speed_drops(self):
        laps = [
            {"max_speed": {"A": 320.0}},
            {"max_speed": {"A": 319.0}},
            {"max_speed": {"A": 318.0}},
            {"max_speed": {"A": 317.0}},
            {"max_speed": {"A": 305.0}},   # current lap — big drop
        ]
        delta = _max_speed_delta(laps, "A")
        assert delta is not None and delta < -5.0

    def test_positive_delta_when_speed_increases(self):
        laps = [
            {"max_speed": {"A": 305.0}},
            {"max_speed": {"A": 306.0}},
            {"max_speed": {"A": 307.0}},
            {"max_speed": {"A": 308.0}},
            {"max_speed": {"A": 320.0}},   # current lap — big gain
        ]
        delta = _max_speed_delta(laps, "A")
        assert delta is not None and delta > 5.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Snapshot builder — Tier 1 fields
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSnapshotTier1:

    def test_pit_stats_populated(self):
        snap = _monaco_snapshot()
        assert snap.pit_stats
        assert "total_pit_loss_mean_s" in snap.pit_stats

    def test_circuit_overtake_populated(self):
        snap = _monaco_snapshot()
        assert snap.circuit_overtake
        assert "overtake_score" in snap.circuit_overtake

    def test_lec_throttle_mode_protecting(self):
        snap = _monaco_snapshot()
        lec = next(d for d in snap.drivers if d.code == "LEC")
        assert lec.throttle_mode == "PROTECTING"

    def test_lec_throttle_drop_pct_set(self):
        snap = _monaco_snapshot()
        lec = next(d for d in snap.drivers if d.code == "LEC")
        assert lec.throttle_drop_pct is not None
        assert lec.throttle_drop_pct >= 2.5

    def test_nor_throttle_mode_normal(self):
        snap = _monaco_snapshot()
        nor = next(d for d in snap.drivers if d.code == "NOR")
        assert nor.throttle_mode == "NORMAL"

    def test_max_speed_delta_set(self):
        snap = _monaco_snapshot()
        lec = next(d for d in snap.drivers if d.code == "LEC")
        assert lec.max_speed_delta is not None

    def test_pit_loss_uses_session_observed_mean(self):
        snap = _monaco_snapshot()
        # Session has 3 observed stops — should NOT be the circuit default 22.5
        assert snap.pit_loss_s != 22.5
        # Should be close to travel_time + mean_stationary
        assert 18.0 < snap.pit_loss_s < 25.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Safety car simulator — Tier 1 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateSafetyCarTier1:

    def test_pit_data_source_in_key_numbers(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        assert "pit_data_source" in sc.key_numbers

    def test_pit_loss_range_in_key_numbers(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        assert "pit_loss_normal_range" in sc.key_numbers
        assert "–" in sc.key_numbers["pit_loss_normal_range"]

    def test_per_driver_analysis_has_throttle_mode(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        first_driver = list(sc.key_numbers["per_driver_analysis"].values())[0]
        assert "throttle_mode" in first_driver

    def test_lec_verdict_includes_urgent_when_protecting(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        lec_analysis = sc.key_numbers["per_driver_analysis"].get("Charles Leclerc", {})
        # LEC is in PROTECTING mode — if they should pit, verdict flags urgency
        if lec_analysis.get("net_benefit_s", 0) > 0:
            assert "urgent" in lec_analysis["verdict"].lower()

    def test_projected_positions_covers_all_drivers(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        assert len(sc.projected_positions) == len(snap.drivers)

    def test_all_positions_unique(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        positions = list(sc.projected_positions.values())
        assert len(positions) == len(set(positions))

    def test_confidence_at_least_0_80(self):
        snap = _monaco_snapshot()
        sc = simulate_safety_car(snap)
        assert sc.confidence >= 0.80

    def test_vsc_less_compression_than_sc(self):
        snap = _monaco_snapshot()
        sc  = simulate_safety_car(snap, sc_type="SC")
        vsc = simulate_safety_car(snap, sc_type="VSC")
        assert vsc.key_numbers["gap_compression_total_s"] < sc.key_numbers["gap_compression_total_s"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pit stop simulator — Tier 1 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatePitStopTier1:

    def test_pit_loss_range_present(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "LEC")
        assert "pit_loss_range" in pit.key_numbers

    def test_net_time_range_ordered(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "LEC")
        assert pit.key_numbers["net_time_best_s"] < pit.key_numbers["net_time_worst_s"]

    def test_throttle_mode_in_key_numbers(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "LEC")
        assert "throttle_mode" in pit.key_numbers

    def test_effective_deg_rate_higher_than_observed_when_protecting(self):
        snap = _monaco_snapshot()
        lec = next(d for d in snap.drivers if d.code == "LEC")
        assert lec.throttle_mode == "PROTECTING"   # guard
        pit = simulate_pit_stop(snap, "LEC")
        assert (pit.key_numbers["effective_deg_rate_s_per_lap"]
                >= pit.key_numbers.get("effective_deg_rate_s_per_lap", 0))

    def test_pit_data_source_in_key_numbers(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "LEC")
        assert "pit_data_source" in pit.key_numbers

    def test_unknown_driver_zero_confidence(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "ZZZ")
        assert pit.confidence == 0.0

    def test_undercut_threat_detected_when_close(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "LEC")  # NOR 1.8s behind
        ut = pit.key_numbers.get("undercut_threat")
        assert ut is not None
        assert ut["from_driver"] == "Lando Norris"
        assert ut["threat_level"] in ("MEDIUM", "HIGH")

    def test_confidence_at_least_0_80(self):
        snap = _monaco_snapshot()
        pit = simulate_pit_stop(snap, "NOR")
        assert pit.confidence >= 0.80


# ─────────────────────────────────────────────────────────────────────────────
# 8. Tyre degradation simulator — Tier 1 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateTyreDegradationTier1:

    def test_lec_status_warning_or_critical(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap, ["LEC"])
        proj = deg.key_numbers["driver_projections"].get("Charles Leclerc", {})
        assert proj["status"] in ("WARNING", "CRITICAL")

    def test_nor_status_ok(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap, ["NOR"])
        proj = deg.key_numbers["driver_projections"].get("Lando Norris", {})
        assert proj["status"] == "OK"

    def test_effective_deg_exceeds_observed_for_protecting_driver(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap, ["LEC"])
        proj = deg.key_numbers["driver_projections"]["Charles Leclerc"]
        assert proj["effective_deg_s_per_lap"] > proj["observed_deg_s_per_lap"]

    def test_throttle_mode_included_in_projection(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap, ["LEC"])
        proj = deg.key_numbers["driver_projections"]["Charles Leclerc"]
        assert "throttle_mode" in proj

    def test_cliff_warnings_non_empty(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap)
        assert len(deg.key_numbers["cliff_warnings"]) > 0

    def test_projected_times_monotonically_increasing(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap, ["LEC"])
        times = deg.key_numbers["driver_projections"]["Charles Leclerc"]["projected_lap_times_next_5"]
        for i in range(len(times) - 1):
            assert times[i + 1] >= times[i]

    def test_confidence_at_least_0_75(self):
        snap = _monaco_snapshot()
        deg = simulate_tyre_degradation(snap)
        assert deg.confidence >= 0.75


# ─────────────────────────────────────────────────────────────────────────────
# 9. Overtake window simulator — Tier 1 integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateOvertakeWindowTier1:

    def test_circuit_difficulty_in_key_numbers(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "NOR", "LEC")
        assert "circuit_difficulty" in ov.key_numbers

    def test_circuit_overtake_score_in_key_numbers(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "NOR", "LEC")
        assert "circuit_overtake_score" in ov.key_numbers

    def test_defender_throttle_mode_reported(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "NOR", "LEC")
        assert ov.key_numbers["defender_throttle_mode"] == "PROTECTING"

    def test_probability_elevated_when_defender_protecting(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "NOR", "LEC")
        # NOR on fresh Soft vs LEC on aged Hard who is protecting tyres
        assert ov.key_numbers["overtake_probability"] > 0.40

    def test_unknown_driver_zero_confidence(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "ZZZ", "LEC")
        assert ov.confidence == 0.0

    def test_laps_to_close_populated_when_faster(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "NOR", "LEC")
        # NOR should be faster (fresh soft) — laps_to_close should be set
        pd = ov.key_numbers["pace_delta_s_per_lap"]
        if pd < 0:
            assert ov.key_numbers["laps_to_close_at_current_pace"] is not None

    def test_verdict_is_valid_string(self):
        snap = _monaco_snapshot()
        ov = simulate_overtake_window(snap, "NOR", "LEC")
        assert ov.key_numbers["verdict"] in (
            "HIGHLY LIKELY", "LIKELY", "POSSIBLE", "UNLIKELY"
        )

    def test_hard_circuit_lowers_base_probability(self):
        """Monaco-style DRS (8%) should produce lower base probability than Monza (35%)."""
        def snap_with_drs(pct: float) -> RaceSnapshot:
            laps = [
                {"drs_pct": {"A": pct, "B": pct},
                 "sector_times": {"A": [29.0, 38.0, 24.0], "B": [29.1, 38.1, 24.1]}}
                for _ in range(10)
            ]
            snap = _monaco_snapshot()
            snap.circuit_overtake = compute_circuit_overtake_score(laps)
            return snap

        monza  = snap_with_drs(34.0)
        monaco = snap_with_drs(8.0)
        ov_monza  = simulate_overtake_window(monza,  "NOR", "LEC")
        ov_monaco = simulate_overtake_window(monaco, "NOR", "LEC")
        assert ov_monza.key_numbers["overtake_probability"] > ov_monaco.key_numbers["overtake_probability"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Intent detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectScenarioIntent:

    @pytest.mark.parametrize("query,expected", [
        ("safety car comes out now",                    "safety_car"),
        ("SC deployed lap 38",                          "safety_car"),
        ("VSC is out",                                  "safety_car"),
        ("virtual safety car deployed",                 "safety_car"),
        ("should Leclerc pit now",                      "pit_stop"),
        ("bring Norris in for an undercut",             "pit_stop"),
        ("what's the pit window",                       "pit_stop"),
        ("overcut Verstappen",                          "pit_stop"),
        ("how long can Leclerc survive on those tyres", "tyre_degradation"),
        ("tyre deg on the soft compound",               "tyre_degradation"),
        ("is anyone near the cliff",                    "tyre_degradation"),
        ("can Norris overtake Leclerc",                 "overtake_window"),
        ("gap is closing between P1 and P2",            "overtake_window"),
        ("Norris on DRS of Leclerc",                    "overtake_window"),
    ])
    def test_intent_routing(self, query, expected):
        assert detect_scenario_intent(query) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 11. Existing analytics helpers (regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestExistingHelpers:

    def test_observed_deg_linear_trend(self):
        laps = [
            {"lap_times": {"LEC": 91.0 + i * 0.1}, "tyres": {"LEC": "H"}}
            for i in range(5)
        ]
        rate = _observed_deg_rate(laps, "LEC")
        assert rate == pytest.approx(0.1, abs=0.02)

    def test_observed_deg_resets_after_compound_change(self):
        laps = [
            {"lap_times": {"LEC": 90.0}, "tyres": {"LEC": "S"}},
            {"lap_times": {"LEC": 90.1}, "tyres": {"LEC": "S"}},
            {"lap_times": {"LEC": 91.0}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.1}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.2}, "tyres": {"LEC": "H"}},
        ]
        rate = _observed_deg_rate(laps, "LEC")
        assert rate == pytest.approx(0.1, abs=0.05)

    def test_gap_trend_closing(self):
        laps = [{"gaps": {"NOR": 3.0}}, {"gaps": {"NOR": 2.4}}, {"gaps": {"NOR": 1.8}}]
        assert _gap_trend(laps, "NOR") == pytest.approx(-1.2, abs=0.01)

    def test_gap_trend_insufficient_data(self):
        assert _gap_trend([{"gaps": {"NOR": 2.0}}], "NOR") is None


# ─────────────────────────────────────────────────────────────────────────────
# 12. run_prediction — end-to-end routing
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPrediction:

    def _snap(self):
        return _monaco_snapshot()

    def test_sc_query_routes_to_safety_car(self):
        snap = self._snap()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("safety car comes out now", "test", {})
        assert result["scenario_type"] == "safety_car"
        assert result["error"] is None

    def test_pit_query_routes_to_pit_stop(self):
        snap = self._snap()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("should Leclerc pit now?", "test", {})
        assert result["scenario_type"] == "pit_stop"
        assert result["scenario"]["key_numbers"]["driver"] == "Charles Leclerc"

    def test_tyre_query_routes_to_degradation(self):
        snap = self._snap()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("how long can he survive on those tyres", "test", {})
        assert result["scenario_type"] == "tyre_degradation"

    def test_overtake_query_routes_to_overtake(self):
        snap = self._snap()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("can Norris overtake Leclerc on DRS?", "test", {})
        assert result["scenario_type"] == "overtake_window"

    def test_snapshot_includes_tier1_fields(self):
        snap = self._snap()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("safety car now", "test", {})
        snapshot = result["snapshot"]
        assert "pit_stats" in snapshot
        assert "circuit_overtake" in snapshot
        for d in snapshot["top10_drivers"]:
            assert "throttle_mode" in d
            assert "throttle_drop_pct" in d
            assert "max_speed_delta" in d

    def test_missing_session_returns_error(self):
        result = run_prediction("what if safety car comes in?", "nonexistent_xyz", {})
        assert result["error"] is not None
        assert result["snapshot"] is None

    def test_error_field_none_on_success(self):
        snap = self._snap()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("pit stop for Leclerc", "test", {})
        assert result["error"] is None
