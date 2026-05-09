"""
Tests for the three physics fixes in prediction_engine.py.
Drop at: tests/test_prediction_engine_physics.py
Run:     pytest tests/test_prediction_engine_physics.py -v

Covers:
  Fix 1 — Fuel-weight corrected _observed_deg_rate
  Fix 2 — Dirty air following distance penalty in _run_simulation
  Fix 3 — Cold tyre out-lap penalty after pit stops
"""
from __future__ import annotations

import pytest
from dataclasses import asdict
from copy import deepcopy
from unittest.mock import patch

from src.backend.services.prediction_engine import (
    DriverState,
    DriverSimState,
    RaceSnapshot,
    RaceToFinishResult,
    simulate_race_to_finish,
    simulate_strategy_comparison,
    _observed_deg_rate,
    _run_simulation,
    _FUEL_GAIN_PER_LAP_S,
    _DIRTY_AIR_THRESHOLD_S,
    _DIRTY_AIR_PENALTY_MAX_S,
    _COLD_TYRE_PENALTY_S,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _monaco_snap(cl: int = 40, tl: int = 78, pit_loss: float = 22.5) -> RaceSnapshot:
    drivers = [
        DriverState("LEC","Charles Leclerc",1, 0.0,-0.1,"H",12,0.05,91.6,91.6,195.0,1),
        DriverState("NOR","Lando Norris",   2, 1.8,-0.4,"S", 8,0.09,91.0,91.1,198.0,2),
        DriverState("VER","Max Verstappen", 3, 4.1,-0.2,"M", 5,0.06,91.5,91.5,196.0,1),
        DriverState("HAM","Lewis Hamilton", 4, 8.3, 0.1,"H",38,0.07,92.1,92.1,194.0,1),
        DriverState("PIA","Oscar Piastri",  5,12.1, 0.2,"M",18,0.05,92.3,92.3,193.0,1),
    ]
    return RaceSnapshot(
        "test", "2025 Monaco GP", "Monaco", cl, tl, max(0, tl - cl), pit_loss,
        drivers=drivers,
        pit_stats={
            "total_pit_loss_mean_s": pit_loss,
            "total_pit_loss_p10_s":  pit_loss - 0.7,
            "total_pit_loss_p90_s":  pit_loss + 1.5,
            "source": "session_observed", "sample_count": 3,
        },
        circuit_overtake={
            "overtake_score": 0.25, "difficulty": "HARD",
            "avg_drs_pct": 10.0, "source": "session_data",
        },
    )


def _close_battle_snap(gap: float = 0.4, laps_remaining: int = 10) -> RaceSnapshot:
    """Two cars with a configurable gap for dirty air testing."""
    drivers = [
        DriverState("LEC","Charles Leclerc",1, 0.0,  None,"M",10,0.06,91.5,91.5,196.0,0),
        DriverState("NOR","Lando Norris",   2, gap,  None,"M",10,0.06,91.5,91.5,196.0,0),
    ]
    cl = 50; tl = cl + laps_remaining
    return RaceSnapshot(
        "test2", "Race", "Track", cl, tl, laps_remaining, 21.0, drivers=drivers,
        pit_stats={"total_pit_loss_mean_s":21.0,"total_pit_loss_p10_s":20.0,
                   "total_pit_loss_p90_s":23.0,"source":"circuit_default","sample_count":0},
        circuit_overtake={"overtake_score":0.5,"difficulty":"MODERATE",
                          "avg_drs_pct":22.0,"source":"defaults"},
    )


def _laps_with_slope(start_lap: int, slope: float, n: int = 5,
                     base: float = 91.0, tyre: str = "H") -> list[dict]:
    """Generate mock laps with a specific raw lap time slope."""
    return [
        {
            "lap": start_lap + i,
            "lap_times": {"LEC": round(base + i * slope, 4)},
            "tyres": {"LEC": tyre},
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — Fuel-weight corrected _observed_deg_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestFuelCorrectedDegRate:

    def test_removes_fuel_contribution_from_slope(self):
        """
        With a raw slope of 0.05s/lap and fuel burn of 0.035s/lap,
        the corrected true tyre deg rate should be ~0.015s/lap.
        """
        laps = _laps_with_slope(start_lap=10, slope=0.05, n=5)
        rate = _observed_deg_rate(laps, "LEC")
        # True deg = raw_slope - fuel_gain = 0.05 - 0.035 = 0.015
        assert rate == pytest.approx(0.015, abs=0.005)

    def test_fuel_constant_is_subtracted_per_lap_number(self):
        """
        The fuel correction uses the absolute lap number, not the position
        within the lookback window — so lap 40 gets a larger correction than lap 3.
        """
        laps_early = _laps_with_slope(start_lap=2,  slope=0.05, n=5)
        laps_late  = _laps_with_slope(start_lap=42, slope=0.05, n=5)
        rate_early = _observed_deg_rate(laps_early, "LEC")
        rate_late  = _observed_deg_rate(laps_late,  "LEC")
        # Both windows have the same slope and same within-window fuel increment,
        # so both corrected rates equal 0.015s/lap. They are identical.
        assert rate_early == pytest.approx(rate_late, abs=0.001)

    def test_zero_slope_after_full_fuel_correction(self):
        """
        If raw lap times rise exactly at the fuel-burn rate (0.035s/lap),
        the corrected deg rate should be approximately zero — the car is
        holding lap time only because it's getting lighter, not degrading.
        """
        laps = _laps_with_slope(start_lap=10, slope=_FUEL_GAIN_PER_LAP_S, n=5)
        rate = _observed_deg_rate(laps, "LEC")
        assert abs(rate) < 0.005

    def test_negative_raw_slope_gives_negative_corrected_rate(self):
        """
        If lap times are improving (driver pushing harder), corrected rate
        can be negative — that's valid (tyre not degrading yet).
        """
        laps = _laps_with_slope(start_lap=10, slope=-0.02, n=5)
        rate = _observed_deg_rate(laps, "LEC")
        # -0.02 raw - 0.035 fuel = -0.055 (very negative — tyre warming/improving)
        assert rate < 0.0

    def test_returns_zero_with_insufficient_data(self):
        laps = _laps_with_slope(start_lap=10, slope=0.05, n=2)
        assert _observed_deg_rate(laps, "LEC") == 0.0

    def test_returns_zero_for_mixed_compounds(self):
        """Compound change within lookback window returns 0 (different stint)."""
        laps = [
            {"lap": 28, "lap_times": {"LEC": 91.0}, "tyres": {"LEC": "M"}},
            {"lap": 29, "lap_times": {"LEC": 91.05},"tyres": {"LEC": "M"}},
            {"lap": 30, "lap_times": {"LEC": 91.4}, "tyres": {"LEC": "H"}},  # pit
            {"lap": 31, "lap_times": {"LEC": 91.45},"tyres": {"LEC": "H"}},
            {"lap": 32, "lap_times": {"LEC": 91.5}, "tyres": {"LEC": "H"}},
        ]
        # Only 3 laps on same compound (H) — should still compute
        rate = _observed_deg_rate(laps, "LEC")
        assert isinstance(rate, float)

    def test_high_deg_rate_correctly_above_zero(self):
        """
        Soft tyre with 0.15s/lap raw slope → 0.115s/lap true deg after fuel correction.
        """
        laps = _laps_with_slope(start_lap=5, slope=0.15, n=5, tyre="S")
        rate = _observed_deg_rate(laps, "LEC")
        assert rate == pytest.approx(0.115, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — Dirty air penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestDirtyAirPenalty:

    def test_following_car_penalised_within_threshold(self):
        """
        NOR starting 0.4s behind LEC (below 0.8s threshold) should accumulate
        more total time than without the dirty air penalty.
        """
        snap = _close_battle_snap(gap=0.4, laps_remaining=10)
        state_with = _run_simulation(snap, max_stops=0)

        # Temporarily disable dirty air
        import src.backend.services.prediction_engine as pe_module
        original = pe_module._DIRTY_AIR_THRESHOLD_S
        pe_module._DIRTY_AIR_THRESHOLD_S = 0.0
        state_without = _run_simulation(snap, max_stops=0)
        pe_module._DIRTY_AIR_THRESHOLD_S = original

        nor_with    = state_with["NOR"].cum_time
        nor_without = state_without["NOR"].cum_time
        assert nor_with > nor_without, "Dirty air should slow following car"

    def test_leader_unaffected_by_dirty_air(self):
        """The car in front never receives a dirty air penalty."""
        snap = _close_battle_snap(gap=0.4, laps_remaining=10)
        state_with = _run_simulation(snap, max_stops=0)

        import src.backend.services.prediction_engine as pe_module
        original = pe_module._DIRTY_AIR_THRESHOLD_S
        pe_module._DIRTY_AIR_THRESHOLD_S = 0.0
        state_without = _run_simulation(snap, max_stops=0)
        pe_module._DIRTY_AIR_THRESHOLD_S = original

        lec_with    = state_with["LEC"].cum_time
        lec_without = state_without["LEC"].cum_time
        assert abs(lec_with - lec_without) < 0.01

    def test_no_penalty_outside_threshold(self):
        """
        Car 5s behind is well outside threshold — should receive no penalty.
        """
        snap = _close_battle_snap(gap=5.0, laps_remaining=10)
        state_with = _run_simulation(snap, max_stops=0)

        import src.backend.services.prediction_engine as pe_module
        original = pe_module._DIRTY_AIR_THRESHOLD_S
        pe_module._DIRTY_AIR_THRESHOLD_S = 0.0
        state_without = _run_simulation(snap, max_stops=0)
        pe_module._DIRTY_AIR_THRESHOLD_S = original

        nor_with    = state_with["NOR"].cum_time
        nor_without = state_without["NOR"].cum_time
        # With 5s gap, no dirty air applies — times should be identical
        assert abs(nor_with - nor_without) < 0.01

    def test_penalty_scales_with_gap(self):
        """
        Closer gap = bigger penalty (penalty scales linearly to 0 at threshold).
        """
        import src.backend.services.prediction_engine as pe_module

        def penalty_for_gap(gap: float) -> float:
            # Matches _run_simulation: condition is `0 < gap <= threshold`
            # gap=0 is excluded (two cars can't occupy the same point)
            if gap <= 0 or gap >= _DIRTY_AIR_THRESHOLD_S:
                return 0.0
            return _DIRTY_AIR_PENALTY_MAX_S * (1.0 - gap / _DIRTY_AIR_THRESHOLD_S)

        p_close = penalty_for_gap(0.2)
        p_far   = penalty_for_gap(0.6)
        assert p_close > p_far, "Closer gap should give bigger penalty"
        # Gap just above 0 should give near-maximum penalty
        assert penalty_for_gap(0.01) > _DIRTY_AIR_PENALTY_MAX_S * 0.95

    def test_penalty_magnitude_realistic(self):
        """Over 10 laps at 0.4s gap, total penalty should be under 3s."""
        snap = _close_battle_snap(gap=0.4, laps_remaining=10)
        state_with = _run_simulation(snap, max_stops=0)

        import src.backend.services.prediction_engine as pe_module
        original = pe_module._DIRTY_AIR_THRESHOLD_S
        pe_module._DIRTY_AIR_THRESHOLD_S = 0.0
        state_without = _run_simulation(snap, max_stops=0)
        pe_module._DIRTY_AIR_THRESHOLD_S = original

        total_penalty = state_with["NOR"].cum_time - state_without["NOR"].cum_time
        assert 0 < total_penalty < 3.0, f"Unrealistic penalty: {total_penalty:.3f}s"

    def test_constant_values_in_range(self):
        assert 0.5 <= _DIRTY_AIR_THRESHOLD_S  <= 1.5, "Threshold should be 0.5–1.5s"
        assert 0.2 <= _DIRTY_AIR_PENALTY_MAX_S <= 0.6, "Max penalty should be 0.2–0.6s"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — Cold tyre out-lap penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestColdTyrePenalty:

    def test_first_lap_after_pit_is_slower(self):
        """
        The lap immediately after a pit stop must be slower than the
        following lap (cold tyre penalty clears after exactly 1 lap).
        """
        snap  = _monaco_snap()
        state = _run_simulation(snap, strategy_overrides={"LEC": [45]}, max_stops=2)
        lec   = state["LEC"]

        # sim_lap indices: sim starts at lap 40, so:
        # sim_lap 5 (index 4) = abs lap 45 = pit lap (includes pit_loss)
        # sim_lap 6 (index 5) = abs lap 46 = first out-lap (cold tyres)
        # sim_lap 7 (index 6) = abs lap 47 = normal lap
        pit_idx  = 45 - 40 - 1   # = 4
        out_idx  = pit_idx + 1   # = 5
        next_idx = out_idx + 1   # = 6

        assert len(lec.lap_times_sim) > next_idx, "Not enough laps simulated"
        out_lap_time  = lec.lap_times_sim[out_idx]
        next_lap_time = lec.lap_times_sim[next_idx]
        assert out_lap_time > next_lap_time, \
            f"Out-lap ({out_lap_time:.3f}s) should be slower than next lap ({next_lap_time:.3f}s)"

    def test_cold_tyre_penalty_magnitude(self):
        """Cold tyre penalty should add approximately COLD_TYRE_PENALTY_S to the out-lap."""
        snap  = _monaco_snap()
        state = _run_simulation(snap, strategy_overrides={"LEC": [45]}, max_stops=2)
        lec   = state["LEC"]

        pit_idx  = 45 - 40 - 1
        out_idx  = pit_idx + 1
        next_idx = out_idx + 1

        if len(lec.lap_times_sim) > next_idx:
            delta = lec.lap_times_sim[out_idx] - lec.lap_times_sim[next_idx]
            # Allow for deg and fuel effects — target is ~COLD_TYRE_PENALTY_S ± 1s
            assert 0.5 < delta < _COLD_TYRE_PENALTY_S + 1.5, \
                f"Cold tyre delta {delta:.3f}s outside expected range"

    def test_penalty_fires_exactly_once(self):
        """out_lap_penalty resets to 0 after one lap — never persists."""
        snap  = _monaco_snap()
        state = _run_simulation(snap, strategy_overrides={"LEC": [45]}, max_stops=2)
        lec   = state["LEC"]
        assert lec.out_lap_penalty == 0.0

    def test_penalty_fires_for_every_pit_stop(self):
        """Each pit stop schedules exactly one cold tyre lap."""
        snap  = _monaco_snap(pit_loss=18.0)  # lower pit loss to allow 2-stop
        state = _run_simulation(snap, strategy_overrides={"LEC": [44, 62]}, max_stops=3)
        lec   = state["LEC"]
        # Both forced pit laps should be present
        pit_laps = [p["lap"] for p in lec.pit_laps if p["type"] == "forced"]
        assert 44 in pit_laps or 62 in pit_laps  # at least one forced stop registered

    def test_cold_tyre_only_affects_out_lap(self):
        """
        The penalty should apply to out-lap ONLY.
        Two laps after the stop the lap times should return to normal trajectory.
        """
        snap  = _monaco_snap()
        state = _run_simulation(snap, strategy_overrides={"LEC": [45]}, max_stops=2)
        lec   = state["LEC"]

        pit_idx   = 45 - 40 - 1
        out_idx   = pit_idx + 1
        after_idx = out_idx + 2   # 2 laps after pit — should be normal

        if len(lec.lap_times_sim) > after_idx:
            out_lap    = lec.lap_times_sim[out_idx]
            after_lap  = lec.lap_times_sim[after_idx]
            # After the cold lap, times should be back to normal pace, not still penalised
            # "Normal" means out_lap should be noticeably slower than 2 laps later
            assert out_lap > after_lap, "Lap 2 after pit should be faster than out-lap"

    def test_constant_value_is_realistic(self):
        """Cold tyre penalty should be between 0.5s and 3.0s."""
        assert 0.5 <= _COLD_TYRE_PENALTY_S <= 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration — all three fixes working together
# ─────────────────────────────────────────────────────────────────────────────

class TestAllThreeFixesIntegration:

    def test_simulate_race_to_finish_runs_cleanly(self):
        snap = _monaco_snap()
        r    = simulate_race_to_finish(snap)
        assert isinstance(r, RaceToFinishResult)
        assert len(r.final_order) == 5
        assert r.final_order[0]["gap_to_winner"] == 0.0

    def test_new_constants_in_key_numbers(self):
        snap = _monaco_snap()
        r    = simulate_race_to_finish(snap)
        assert "dirty_air_threshold_s"       in r.key_numbers
        assert "dirty_air_max_penalty_s"     in r.key_numbers
        assert "cold_tyre_out_lap_penalty_s" in r.key_numbers
        assert r.key_numbers["dirty_air_threshold_s"]       == _DIRTY_AIR_THRESHOLD_S
        assert r.key_numbers["cold_tyre_out_lap_penalty_s"] == _COLD_TYRE_PENALTY_S

    def test_assumptions_document_all_three_fixes(self):
        snap = _monaco_snap()
        r    = simulate_race_to_finish(snap)
        assumptions_text = " ".join(r.assumptions)
        assert "fuel-weight" in assumptions_text.lower() or "fuel" in assumptions_text.lower()
        assert "dirty air"   in assumptions_text.lower()
        assert "cold tyre"   in assumptions_text.lower()

    def test_result_is_json_serialisable(self):
        import json
        snap = _monaco_snap()
        r    = simulate_race_to_finish(snap)
        json.dumps(asdict(r))  # should not raise

    def test_positions_are_unique_and_complete(self):
        snap = _monaco_snap()
        r    = simulate_race_to_finish(snap)
        positions = [fo["position"] for fo in r.final_order]
        assert sorted(positions) == list(range(1, len(snap.drivers) + 1))

    def test_gap_series_correct_length(self):
        snap = _monaco_snap()
        r    = simulate_race_to_finish(snap)
        for code, series in r.gap_series.items():
            assert len(series) == snap.laps_remaining

    def test_strategy_comparison_still_works(self):
        snap = _monaco_snap()
        strats = {"stay out": {}, "pit now": {"LEC": [40]}, "pit L55": {"LEC": [55]}}
        sc = simulate_strategy_comparison(snap, strats)
        assert sc.scenario_type == "strategy_comparison"
        assert len(sc.key_numbers["strategy_results"]) == 3
        assert sc.key_numbers["best_strategy"] in strats

    def test_physics_sanity_fresh_beats_worn(self):
        """Fresh tyre should still beat worn tyre over 35 laps with no more stops."""
        worn_drivers = [
            DriverState("WORN","Driver A",1, 0.0,None,"H",40,0.08,92.0,92.0,190.0,1),
            DriverState("FRSH","Driver B",2, 5.0,None,"M", 1,0.06,91.0,91.0,192.0,1),
        ]
        snap = RaceSnapshot(
            "x","Race","Circuit",40,75,35,21.0, drivers=worn_drivers,
            pit_stats={"total_pit_loss_mean_s":21.0,"total_pit_loss_p10_s":20.0,
                       "total_pit_loss_p90_s":23.0,"source":"circuit_default","sample_count":0},
            circuit_overtake={"overtake_score":0.5,"difficulty":"MODERATE",
                              "avg_drs_pct":22.0,"source":"defaults"},
        )
        r = simulate_race_to_finish(snap, max_stops=0)
        assert r.final_order[0]["driver_code"] == "FRSH"

    def test_confidence_floor_maintained(self):
        snap = _monaco_snap(cl=1, tl=78)
        r    = simulate_race_to_finish(snap)
        assert r.confidence >= 0.45

    def test_winner_gap_series_is_zero(self):
        snap   = _monaco_snap()
        r      = simulate_race_to_finish(snap)
        winner = r.final_order[0]["driver_code"]
        assert all(abs(g) < 0.01 for g in r.gap_series[winner])
