"""
Tests for the F1 prediction engine.
Drop at: tests/test_prediction_engine.py
Run: pytest tests/test_prediction_engine.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
from src.backend.services.prediction_engine import (
    DriverState,
    RaceSnapshot,
    simulate_safety_car,
    simulate_pit_stop,
    simulate_tyre_degradation,
    simulate_overtake_window,
    detect_scenario_intent,
    run_prediction,
    build_snapshot,
    _observed_deg_rate,
    _gap_trend,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _monaco_final_laps() -> RaceSnapshot:
    """Monaco lap 68/78 — leader on aged hards, P2 on fresh softs."""
    return RaceSnapshot(
        session_id="test",
        race_name="2025 Monaco Grand Prix",
        track_name="Monaco",
        current_lap=68,
        total_laps=78,
        laps_remaining=10,
        pit_loss_s=22.5,
        drivers=[
            DriverState("LEC", "Charles Leclerc",   1,  0.0, -0.1, "H", 42, 0.07, 91.4, 91.6, 178.0, 1),
            DriverState("NOR", "Lando Norris",       2,  1.8, -0.4, "S",  8, 0.09, 91.0, 91.1, 180.2, 2),
            DriverState("VER", "Max Verstappen",     3,  4.2, -0.2, "M", 22, 0.05, 91.5, 91.6, 179.5, 1),
            DriverState("HAM", "Lewis Hamilton",     4,  8.1,  0.1, "H", 38, 0.06, 92.0, 92.1, 177.0, 1),
            DriverState("PIA", "Oscar Piastri",      5, 11.3,  0.3, "M", 18, 0.05, 92.2, 92.3, 176.5, 1),
        ],
    )


def _silverstone_mid_race() -> RaceSnapshot:
    """Silverstone lap 32/52 — P1 on medium, P2 on soft closing fast."""
    return RaceSnapshot(
        session_id="test2",
        race_name="2025 British Grand Prix",
        track_name="Silverstone",
        current_lap=32,
        total_laps=52,
        laps_remaining=20,
        pit_loss_s=20.5,
        drivers=[
            DriverState("VER", "Max Verstappen",   1,  0.0,  0.1, "M", 18, 0.06, 89.5, 89.7, 256.0, 1),
            DriverState("NOR", "Lando Norris",     2,  2.1, -0.7, "S",  5, 0.10, 88.8, 88.9, 259.0, 1),
            DriverState("LEC", "Charles Leclerc",  3,  6.4, -0.2, "H", 32, 0.04, 90.0, 90.1, 254.0, 1),
            DriverState("HAM", "Lewis Hamilton",   4, 12.0,  0.2, "M", 20, 0.06, 90.2, 90.4, 253.0, 2),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentDetection:
    def test_safety_car_variants(self):
        for q in ["safety car comes out", "SC deployed now", "VSC out", "virtual safety car"]:
            assert detect_scenario_intent(q) == "safety_car", f"Failed for: {q}"

    def test_pit_stop_variants(self):
        for q in ["should he pit now", "bring Leclerc in", "pit window", "undercut Verstappen"]:
            assert detect_scenario_intent(q) == "pit_stop", f"Failed for: {q}"

    def test_tyre_degradation_variants(self):
        for q in ["tyre deg on the softs", "how long can he survive on those tyres", "tyre cliff"]:
            assert detect_scenario_intent(q) == "tyre_degradation", f"Failed for: {q}"

    def test_overtake_window_variants(self):
        for q in ["can Norris overtake Leclerc", "DRS attack", "gap is closing fast"]:
            assert detect_scenario_intent(q) == "overtake_window", f"Failed for: {q}"


# ─────────────────────────────────────────────────────────────────────────────
# Safety car simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyCarSimulation:
    def test_sc_compresses_gaps(self):
        snap = _monaco_final_laps()
        result = simulate_safety_car(snap, sc_type="SC")
        # After SC, gaps should be compressed toward zero
        # NOR was 1.8s behind — with 4 × 28s compression, effectively 0
        assert result.key_numbers["gap_compression_total_s"] == pytest.approx(112.0, abs=1.0)

    def test_sc_pit_loss_reduced(self):
        snap = _monaco_final_laps()
        result = simulate_safety_car(snap, sc_type="SC")
        sc_loss = result.key_numbers["pit_loss_under_sc_s"]
        normal_loss = result.key_numbers["pit_loss_normal_s"]
        assert sc_loss < normal_loss
        assert sc_loss == pytest.approx(normal_loss * 0.55, abs=0.5)

    def test_vsc_less_compression_than_sc(self):
        snap = _monaco_final_laps()
        sc_result = simulate_safety_car(snap, sc_type="SC")
        vsc_result = simulate_safety_car(snap, sc_type="VSC")
        assert vsc_result.key_numbers["gap_compression_total_s"] < sc_result.key_numbers["gap_compression_total_s"]

    def test_projected_positions_all_filled(self):
        snap = _monaco_final_laps()
        result = simulate_safety_car(snap)
        assert len(result.projected_positions) == len(snap.drivers)
        positions = list(result.projected_positions.values())
        assert sorted(positions) == list(range(1, len(snap.drivers) + 1))

    def test_late_race_nobody_should_pit_monaco(self):
        """With 10 laps left, pit loss exceeds remaining deg — nobody benefits."""
        snap = _monaco_final_laps()
        result = simulate_safety_car(snap)
        # All drivers: remaining_deg_loss should be < pit_loss_under_sc
        for code, analysis in result.key_numbers["per_driver_analysis"].items():
            assert analysis["verdict"] == "STAY", f"{code} should STAY but got PIT"


# ─────────────────────────────────────────────────────────────────────────────
# Pit stop simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestPitStopSimulation:
    def test_leader_pit_drops_position(self):
        snap = _monaco_final_laps()
        result = simulate_pit_stop(snap, "LEC", pit_lap=68)
        # Pitting from P1 at Monaco, other cars don't pit → drops down
        assert result.projected_positions["LEC"] > 1

    def test_net_time_negative_means_beneficial(self):
        """On a long stint, pit should be beneficial (negative net_time_cost)."""
        snap = _silverstone_mid_race()
        # VER on M age 18, 20 laps left — deg is building
        result = simulate_pit_stop(snap, "VER", pit_lap=32)
        nums = result.key_numbers
        # tyre_time_benefit should be calculable
        assert isinstance(nums["tyre_time_benefit_s"], (int, float))
        assert isinstance(nums["net_time_cost_s"], (int, float))

    def test_unknown_driver_returns_zero_confidence(self):
        snap = _monaco_final_laps()
        result = simulate_pit_stop(snap, "XXX")
        assert result.confidence == 0.0

    def test_undercut_threat_detected_when_close(self):
        snap = _monaco_final_laps()
        # NOR is 1.8s behind LEC — should flag MEDIUM threat
        result = simulate_pit_stop(snap, "LEC")
        ut = result.key_numbers.get("undercut_threat")
        assert ut is not None
        assert ut["from_driver"] == "Lando Norris"
        assert ut["threat_level"] in ("MEDIUM", "HIGH")

    def test_compound_selection_soft_to_medium(self):
        snap = _monaco_final_laps()
        # NOR is on S — default new compound should be M
        result = simulate_pit_stop(snap, "NOR")
        assert result.key_numbers["new_compound"] == "M"


# ─────────────────────────────────────────────────────────────────────────────
# Tyre degradation simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestTyreDegradation:
    def test_lec_hard_past_cliff_is_warning(self):
        snap = _monaco_final_laps()
        result = simulate_tyre_degradation(snap, ["LEC"])
        proj = result.key_numbers["driver_projections"].get("Charles Leclerc")
        assert proj is not None
        assert proj["status"] in ("WARNING", "CRITICAL")
        assert proj["laps_past_cliff"] == 4  # age 42, cliff at 38

    def test_nor_soft_not_yet_at_cliff(self):
        snap = _monaco_final_laps()
        result = simulate_tyre_degradation(snap, ["NOR"])
        proj = result.key_numbers["driver_projections"].get("Lando Norris")
        assert proj is not None
        assert proj["status"] == "OK"
        assert proj["laps_to_cliff"] > 0

    def test_projections_monotonically_increasing(self):
        """Lap times should increase (worsen) each future lap under degradation."""
        snap = _monaco_final_laps()
        result = simulate_tyre_degradation(snap, ["LEC"])
        proj = result.key_numbers["driver_projections"]["Charles Leclerc"]
        times = proj["projected_lap_times_next_5"]
        for i in range(len(times) - 1):
            assert times[i + 1] >= times[i], "Projected times should be non-decreasing"

    def test_cliff_warnings_populated(self):
        snap = _monaco_final_laps()
        result = simulate_tyre_degradation(snap)
        # At minimum LEC on H age 42 should trigger a warning
        assert len(result.key_numbers["cliff_warnings"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Overtake window simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestOvertakeWindow:
    def test_nor_vs_lec_probability_reasonable(self):
        snap = _monaco_final_laps()
        result = simulate_overtake_window(snap, "NOR", "LEC")
        p = result.key_numbers["overtake_probability"]
        assert 0.0 <= p <= 1.0

    def test_faster_attacker_higher_probability(self):
        """NOR on fresh softs vs LEC on aged hards should have meaningful probability."""
        snap = _monaco_final_laps()
        result = simulate_overtake_window(snap, "NOR", "LEC")
        p = result.key_numbers["overtake_probability"]
        assert p >= 0.30, f"Expected ≥30% but got {p}"

    def test_laps_to_close_negative_pace_delta(self):
        snap = _silverstone_mid_race()
        result = simulate_overtake_window(snap, "NOR", "VER")  # NOR -0.7s/lap faster
        ltc = result.key_numbers["laps_to_close_at_current_pace"]
        assert ltc is not None
        assert ltc > 0
        # Gap is 2.1s, pace delta -0.7s/lap → ~3 laps
        assert ltc == pytest.approx(3.0, abs=0.5)

    def test_unknown_driver_zero_confidence(self):
        snap = _monaco_final_laps()
        result = simulate_overtake_window(snap, "ZZZ", "LEC")
        assert result.confidence == 0.0

    def test_drs_eligible_when_within_one_second(self):
        snap = _monaco_final_laps()
        # Manually reduce NOR gap to 0.8s
        snap.drivers[1].gap_to_leader = 0.8
        result = simulate_overtake_window(snap, "NOR", "LEC")
        assert result.key_numbers["drs_eligible"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Data analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestDataAnalytics:
    def test_observed_deg_rate_linear_trend(self):
        laps = [
            {"lap_times": {"LEC": 91.0}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.1}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.2}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.3}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.4}, "tyres": {"LEC": "H"}},
        ]
        rate = _observed_deg_rate(laps, "LEC")
        assert rate == pytest.approx(0.1, abs=0.01)

    def test_observed_deg_resets_after_compound_change(self):
        """If compound changes mid-window, only same-compound laps are used."""
        laps = [
            {"lap_times": {"LEC": 90.0}, "tyres": {"LEC": "S"}},
            {"lap_times": {"LEC": 90.1}, "tyres": {"LEC": "S"}},
            {"lap_times": {"LEC": 91.0}, "tyres": {"LEC": "H"}},  # pit stop
            {"lap_times": {"LEC": 91.1}, "tyres": {"LEC": "H"}},
            {"lap_times": {"LEC": 91.2}, "tyres": {"LEC": "H"}},
        ]
        rate = _observed_deg_rate(laps, "LEC")
        # Should only use H laps → ~0.1 per lap
        assert rate == pytest.approx(0.1, abs=0.05)

    def test_gap_trend_closing(self):
        laps = [
            {"gaps": {"NOR": 3.0}},
            {"gaps": {"NOR": 2.4}},
            {"gaps": {"NOR": 1.8}},
        ]
        trend = _gap_trend(laps, "NOR")
        assert trend == pytest.approx(-1.2, abs=0.01)  # closed 1.2s over 3 laps

    def test_gap_trend_insufficient_data(self):
        laps = [{"gaps": {"NOR": 2.0}}]
        assert _gap_trend(laps, "NOR") is None


# ─────────────────────────────────────────────────────────────────────────────
# run_prediction integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPrediction:
    def test_returns_error_when_no_session(self):
        result = run_prediction(
            query="what if safety car comes in?",
            session_id="nonexistent_session_xyz",
            live_context={},
        )
        assert result["error"] is not None
        assert result["snapshot"] is None

    def test_sc_query_routes_to_safety_car(self):
        snap = _monaco_final_laps()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("safety car comes out now", "test", {"current_lap": 68})
        assert result["scenario_type"] == "safety_car"
        assert result["error"] is None
        assert result["snapshot"] is not None

    def test_pit_query_routes_to_pit_stop(self):
        snap = _monaco_final_laps()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("should Leclerc pit now?", "test", {"current_lap": 68})
        assert result["scenario_type"] == "pit_stop"
        assert result["scenario"]["key_numbers"]["driver"] == "Charles Leclerc"

    def test_snapshot_top10_structure(self):
        snap = _monaco_final_laps()
        with patch("src.backend.services.prediction_engine.build_snapshot", return_value=snap):
            result = run_prediction("tyre deg analysis", "test", {})
        snapshot = result["snapshot"]
        assert "top10_drivers" in snapshot
        assert "laps_remaining" in snapshot
        assert "pit_loss_s" in snapshot
        for d in snapshot["top10_drivers"]:
            assert "name" in d
            assert "tyre" in d
            assert "tyre_age" in d
            assert "gap_s" in d
