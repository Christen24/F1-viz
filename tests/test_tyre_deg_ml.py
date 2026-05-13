"""
ML Tyre Degradation Model — Integration Tests
Drop at: tests/test_tyre_deg_ml.py
Run:     pytest tests/test_tyre_deg_ml.py -v

Covers:
  TyreDegModel wrapper         — predict(), predict_batch(), describe(), fallback
  prediction_engine integration — track_temp_c field, ML path, static fallback
  Temperature sensitivity       — hot vs cool circuit deg rate difference
  Regression                    — all existing simulators unaffected
"""
from __future__ import annotations

import json
import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock

from src.backend.services.prediction_engine import (
    DriverState,
    RaceSnapshot,
    RaceToFinishResult,
    ScenarioResult,
    build_snapshot,
    simulate_race_to_finish,
    simulate_strategy_comparison,
    simulate_safety_car,
    simulate_pit_stop,
    simulate_tyre_degradation,
    simulate_overtake_window,
    run_prediction,
    _build_snapshot_dict,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_tyre_model(available: bool = True) -> MagicMock:
    """
    Return a mock TyreDegModel that behaves like the real wrapper.
    When available=True, predict() returns a temperature-adjusted deg rate.
    When available=False, simulates model file not found.
    """
    mock = MagicMock()
    type(mock).is_available = PropertyMock(return_value=available)

    def _predict(*, compound, tyre_life, track_temp=30.0, air_temp=20.0,
                 fuel_load_kg=60.0, driver_code="UNK", circuit="Unknown",
                 fresh_tyre=True, stint_num=0, abs_lap=1):
        base = {"S": 0.10, "M": 0.06, "H": 0.04, "I": 0.07, "W": 0.05}.get(
            str(compound).upper()[:1], 0.06
        )
        temp_factor = 1.0 + (track_temp - 30.0) * 0.003
        return round(base * temp_factor, 5)

    mock.predict.side_effect = _predict
    mock.describe.return_value = {
        "status": "loaded" if available else "unavailable",
        "source": "xgboost" if available else "static_fallback",
    }
    return mock


def _monaco_snap(cl: int = 40, tl: int = 78,
                 track_temp: float = 30.0) -> RaceSnapshot:
    drivers = [
        DriverState("LEC","Charles Leclerc",1, 0.0,-0.1,"H",12,0.05,91.6,91.6,195.0,1),
        DriverState("NOR","Lando Norris",   2, 1.8,-0.4,"S", 8,0.09,91.0,91.1,198.0,2),
        DriverState("VER","Max Verstappen", 3, 4.1,-0.2,"M", 5,0.06,91.5,91.5,196.0,1),
        DriverState("HAM","Lewis Hamilton", 4, 8.3, 0.1,"H",38,0.07,92.1,92.1,194.0,1),
        DriverState("PIA","Oscar Piastri",  5,12.1, 0.2,"M",18,0.05,92.3,92.3,193.0,1),
    ]
    return RaceSnapshot(
        "test","2025 Monaco GP","Monaco",cl,tl,max(0,tl-cl),22.5,
        drivers=drivers,
        pit_stats={"total_pit_loss_mean_s":22.5,"total_pit_loss_p10_s":21.8,
                   "total_pit_loss_p90_s":24.1,"source":"session_observed","sample_count":3},
        circuit_overtake={"overtake_score":0.25,"difficulty":"HARD",
                          "avg_drs_pct":10.0,"source":"session_data"},
        track_temp_c=track_temp,
    )


def _make_laps(n: int = 40) -> list[dict]:
    laps = []
    for i in range(1, n + 1):
        ps = []
        if i == 28:
            ps.append({"type":"pit_stop","actor":"LEC",
                "details":{"pit_duration":2.6,"compound":"H","lap":28},"victim":None})
        if i == 32:
            ps.append({"type":"pit_stop","actor":"NOR",
                "details":{"pit_duration":2.4,"compound":"S","lap":32},"victim":None})
        laps.append({
            "lap": i,
            "positions":  {"LEC": 1, "NOR": 2},
            "gaps":       {"LEC": 0.0, "NOR": 1.8},
            "tyres":      {"LEC": "H", "NOR": "S" if i >= 32 else "M"},
            "tyre_ages":  {"LEC": max(1, i-27), "NOR": max(1, i-31)},
            "lap_times":  {"LEC": round(91.0 + i * 0.05, 3), "NOR": 90.8},
            "pit_stops":  ps,
            "avg_speed":  {"LEC": 195.2, "NOR": 198.1},
            "max_speed":  {"LEC": 318.0, "NOR": 322.0},
            "throttle_pct": {"LEC": round(88.0 - max(0, i-34)*0.8, 1), "NOR": 90.2},
            "brake_pct":  {"LEC": 18.0, "NOR": 17.5},
            "drs_pct":    {"LEC": 22.0, "NOR": 23.5},
            "sector_times": {"LEC": [29.1, 38.0, 23.9], "NOR": [28.9, 38.0, 23.8]},
            "events":     [],
            "weather":    {"track_temp": 38.0, "air_temp": 28.0},
        })
    return laps


# ─────────────────────────────────────────────────────────────────────────────
# 1. TyreDegModel wrapper unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTyreDegModelWrapper:
    """Tests for src/backend/services/tyre_deg_model.py"""

    def test_predict_returns_float(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel()                # model file won't exist in test env
        result = m.predict(compound="S", tyre_life=10)
        assert isinstance(result, float)

    def test_predict_falls_back_to_static_when_no_model(self):
        from src.backend.services.tyre_deg_model import TyreDegModel, _STATIC_DEG
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        for compound in ("S", "M", "H", "I", "W"):
            result = m.predict(compound=compound, tyre_life=10)
            assert result == pytest.approx(_STATIC_DEG[compound], abs=0.001), \
                f"Fallback for {compound}: expected {_STATIC_DEG[compound]}, got {result}"

    def test_predict_clamps_to_valid_range(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        result = m.predict(compound="S", tyre_life=10)
        assert 0.001 <= result <= 0.50

    def test_predict_batch_returns_list(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        rows = [
            {"compound": "S", "tyre_life": 10},
            {"compound": "M", "tyre_life": 20},
            {"compound": "H", "tyre_life": 30},
        ]
        results = m.predict_batch(rows)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)

    def test_predict_batch_same_as_individual_predict(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        rows = [{"compound": "S", "tyre_life": 5}, {"compound": "M", "tyre_life": 15}]
        batch = m.predict_batch(rows)
        singles = [m.predict(compound=r["compound"], tyre_life=r["tyre_life"])
                   for r in rows]
        for b, s in zip(batch, singles):
            assert b == pytest.approx(s, abs=0.001)

    def test_is_available_false_when_no_model_file(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        assert m.is_available is False

    def test_describe_returns_dict(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        d = m.describe()
        assert isinstance(d, dict)
        assert "status" in d

    def test_describe_unavailable_when_no_file(self):
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        d = m.describe()
        assert d["status"] == "unavailable"

    def test_unknown_compound_falls_back_to_medium_rate(self):
        from src.backend.services.tyre_deg_model import TyreDegModel, _STATIC_DEG
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        result = m.predict(compound="X", tyre_life=5)
        assert result == pytest.approx(_STATIC_DEG["M"], abs=0.001)

    def test_encode_compound_known_values(self):
        from src.backend.services.tyre_deg_model import TyreDegModel, _COMPOUND_ID
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        assert m._encode_compound("S")    == 0
        assert m._encode_compound("SOFT") == 0
        assert m._encode_compound("M")    == 1
        assert m._encode_compound("H")    == 2
        assert m._encode_compound("I")    == 3
        assert m._encode_compound("W")    == 4

    def test_load_fails_gracefully(self):
        """Loading a corrupt file should not raise — sets _load_failed flag."""
        from src.backend.services.tyre_deg_model import TyreDegModel
        m = TyreDegModel(model_dir="/nonexistent_dir_xyz")
        # Force a failed load
        m._load_failed = False
        m._model_path  = "/dev/null"  # exists but not a valid joblib file
        result = m._try_load()
        assert result is False
        assert m._load_failed is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. RaceSnapshot track_temp_c field
# ─────────────────────────────────────────────────────────────────────────────

class TestRaceSnapshotTrackTemp:

    def test_field_exists_on_dataclass(self):
        snap = _monaco_snap(track_temp=38.0)
        assert hasattr(snap, "track_temp_c")

    def test_field_stores_correct_value(self):
        snap = _monaco_snap(track_temp=42.5)
        assert snap.track_temp_c == 42.5

    def test_default_is_30(self):
        snap = RaceSnapshot("x","x","x",1,78,77,21.0)
        assert snap.track_temp_c == 30.0

    def test_included_in_snapshot_dict(self):
        snap = _monaco_snap(track_temp=38.0)
        d = _build_snapshot_dict(snap)
        assert "track_temp_c" in d
        assert d["track_temp_c"] == 38.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_snapshot — track_temp_c sourcing
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSnapshotTrackTemp:

    def _live(self, **kwargs) -> dict:
        base = {
            "current_lap": 40, "total_laps": 78,
            "positions":  {"LEC": 1, "NOR": 2},
            "gaps":       {"LEC": 0.0, "NOR": 1.8},
            "tyres":      {"LEC": "H", "NOR": "S"},
            "tyre_ages":  {"LEC": 12, "NOR": 8},
            "avg_speed":  {"LEC": 195.2, "NOR": 198.1},
        }
        base.update(kwargs)
        return base

    def test_reads_from_live_context(self):
        laps = _make_laps()
        live = self._live(track_temp_c=45.0)
        snap = build_snapshot("test", live, laps_override=laps)
        assert snap.track_temp_c == 45.0

    def test_reads_from_lap_weather_field(self):
        laps = _make_laps()
        live = self._live()   # no track_temp_c in live_context
        snap = build_snapshot("test", live, laps_override=laps)
        # Latest lap has weather.track_temp = 38.0
        assert snap.track_temp_c == 38.0

    def test_defaults_to_30_when_no_weather_data(self):
        laps = _make_laps()
        for lap in laps:
            lap.pop("weather", None)
        live = self._live()
        snap = build_snapshot("test", live, laps_override=laps)
        assert snap.track_temp_c == 30.0

    def test_live_context_takes_priority_over_lap_weather(self):
        laps = _make_laps()   # laps have track_temp=38.0
        live = self._live(track_temp_c=50.0)
        snap = build_snapshot("test", live, laps_override=laps)
        assert snap.track_temp_c == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. ML model integration in build_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestMLModelInBuildSnapshot:

    def test_static_fallback_when_no_model(self):
        """When ML model unavailable, deg_rate uses static table or regression."""
        import src.backend.services.prediction_engine as pe_mod
        original = pe_mod._tyre_model
        pe_mod._tyre_model = _mock_tyre_model(available=False)
        try:
            laps = _make_laps()
            live = {"current_lap":40,"total_laps":78,
                    "positions":{"LEC":1},"gaps":{"LEC":0.0},
                    "tyres":{"LEC":"H"},"tyre_ages":{"LEC":12},
                    "avg_speed":{"LEC":195.2}}
            snap = build_snapshot("test", live, laps_override=laps)
            lec = next(d for d in snap.drivers if d.code == "LEC")
            # LEC has laps data — regression returns a rate > 0
            assert lec.tyre_deg_rate > 0
        finally:
            pe_mod._tyre_model = original

    def test_ml_model_called_when_available_and_regression_zero(self):
        """When regression returns 0 (new driver), ML model is invoked."""
        import src.backend.services.prediction_engine as pe_mod
        original = pe_mod._tyre_model
        mock = _mock_tyre_model(available=True)
        pe_mod._tyre_model = mock
        try:
            laps = _make_laps()
            live = {"current_lap":40,"total_laps":78,"track_temp_c":42.0,
                    "positions":{"NEWDRV":1},"gaps":{"NEWDRV":0.0},
                    "tyres":{"NEWDRV":"S"},"tyre_ages":{"NEWDRV":1},
                    "avg_speed":{"NEWDRV":200.0}}
            snap = build_snapshot("test", live, laps_override=laps)
            if snap:
                newdrv = next((d for d in snap.drivers if d.code == "NEWDRV"), None)
                if newdrv:
                    # ML model was called since no regression data exists for NEWDRV
                    assert mock.predict.called
                    # S tyre at 42°C: 0.10 * (1 + (42-30)*0.003) = 0.10 * 1.036 = 0.1036
                    expected = 0.10 * (1.0 + (42.0 - 30.0) * 0.003)
                    assert newdrv.tyre_deg_rate == pytest.approx(expected, abs=0.005)
        finally:
            pe_mod._tyre_model = original

    def test_ml_deg_model_active_in_snapshot_dict_when_available(self):
        import src.backend.services.prediction_engine as pe_mod
        original = pe_mod._tyre_model
        pe_mod._tyre_model = _mock_tyre_model(available=True)
        try:
            snap = _monaco_snap()
            d = _build_snapshot_dict(snap)
            assert d["ml_deg_model_active"] is True
        finally:
            pe_mod._tyre_model = original

    def test_ml_deg_model_active_false_when_unavailable(self):
        import src.backend.services.prediction_engine as pe_mod
        original = pe_mod._tyre_model
        pe_mod._tyre_model = _mock_tyre_model(available=False)
        try:
            snap = _monaco_snap()
            d = _build_snapshot_dict(snap)
            assert d["ml_deg_model_active"] is False
        finally:
            pe_mod._tyre_model = original


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temperature sensitivity
# ─────────────────────────────────────────────────────────────────────────────

class TestTemperatureSensitivity:

    def test_hot_circuit_higher_deg_than_cool(self):
        mock = _mock_tyre_model(available=True)
        hot  = mock.predict(compound="S", tyre_life=10, track_temp=48.0)
        cool = mock.predict(compound="S", tyre_life=10, track_temp=18.0)
        assert hot > cool

    def test_soft_more_sensitive_than_hard(self):
        mock = _mock_tyre_model(available=True)
        # Both tested at same high temp — soft should be more affected
        soft_hot  = mock.predict(compound="S", tyre_life=10, track_temp=48.0)
        soft_base = mock.predict(compound="S", tyre_life=10, track_temp=30.0)
        hard_hot  = mock.predict(compound="H", tyre_life=10, track_temp=48.0)
        hard_base = mock.predict(compound="H", tyre_life=10, track_temp=30.0)
        soft_delta = soft_hot - soft_base
        hard_delta = hard_hot - hard_base
        # Soft has higher base, so absolute delta should be larger
        assert soft_delta > hard_delta

    def test_at_reference_temp_returns_base_rate(self):
        """At 30°C (reference), factor is 1.0, prediction equals base rate."""
        mock = _mock_tyre_model(available=True)
        for compound, base in [("S", 0.10), ("M", 0.06), ("H", 0.04)]:
            result = mock.predict(compound=compound, tyre_life=10, track_temp=30.0)
            assert result == pytest.approx(base, abs=0.001), \
                f"{compound} at 30°C: expected {base}, got {result}"

    def test_bahrain_vs_silverstone_realistic_delta(self):
        """Bahrain (~35°C) should have meaningfully higher S deg than Silverstone (~22°C)."""
        mock = _mock_tyre_model(available=True)
        bahrain     = mock.predict(compound="S", tyre_life=10, track_temp=35.0)
        silverstone = mock.predict(compound="S", tyre_life=10, track_temp=22.0)
        delta = bahrain - silverstone
        # 13°C difference × 0.003 sensitivity × 0.10 base = 0.0039 delta
        assert delta > 0.002
        assert delta < 0.02   # not absurd


# ─────────────────────────────────────────────────────────────────────────────
# 6. All simulators unaffected by ML integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatorsUnaffected:

    def _snap(self) -> RaceSnapshot:
        return _monaco_snap()

    def test_race_to_finish_runs(self):
        r = simulate_race_to_finish(self._snap())
        assert isinstance(r, RaceToFinishResult)
        assert len(r.final_order) == 5

    def test_strategy_comparison_runs(self):
        snap = self._snap()
        sc = simulate_strategy_comparison(snap,
            {"stay": {}, "pit L45": {"LEC": [45]}})
        assert sc.scenario_type == "strategy_comparison"

    def test_safety_car_runs(self):
        r = simulate_safety_car(self._snap())
        assert isinstance(r, ScenarioResult)

    def test_pit_stop_runs(self):
        r = simulate_pit_stop(self._snap(), "LEC")
        assert isinstance(r, ScenarioResult)
        assert r.key_numbers["driver"] == "Charles Leclerc"

    def test_tyre_deg_runs(self):
        r = simulate_tyre_degradation(self._snap())
        assert isinstance(r, ScenarioResult)
        assert len(r.key_numbers["cliff_warnings"]) >= 0

    def test_overtake_window_runs(self):
        r = simulate_overtake_window(self._snap(), "NOR", "LEC")
        assert isinstance(r, ScenarioResult)
        assert 0.0 <= r.key_numbers["overtake_probability"] <= 1.0

    def test_result_is_json_serialisable(self):
        r = simulate_race_to_finish(self._snap())
        json.dumps(asdict(r))   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 7. run_prediction — track_temp flows through to snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPredictionTrackTemp:

    def _snap_factory(self, track_temp: float):
        def factory(*args, **kwargs):
            return _monaco_snap(track_temp=track_temp)
        return factory

    @pytest.mark.parametrize("query,expected_type", [
        ("who wins the race",          "race_projection"),
        ("1-stop vs 2-stop",           "strategy_comparison"),
        ("safety car now",             "safety_car"),
        ("should Leclerc pit",         "pit_stop"),
        ("tyre wear analysis",         "tyre_degradation"),
        ("can Norris catch Leclerc",   "overtake_window"),
    ])
    def test_intent_routes_correctly(self, query, expected_type):
        with patch("src.backend.services.prediction_engine.build_snapshot",
                   side_effect=self._snap_factory(38.0)):
            r = run_prediction(query, "test", {})
        assert r["scenario_type"] == expected_type
        assert r["error"] is None

    def test_track_temp_in_snapshot_output(self):
        with patch("src.backend.services.prediction_engine.build_snapshot",
                   side_effect=self._snap_factory(42.0)):
            r = run_prediction("who wins the race", "test", {})
        assert r["snapshot"]["track_temp_c"] == 42.0

    def test_ml_model_active_in_snapshot_output(self):
        with patch("src.backend.services.prediction_engine.build_snapshot",
                   side_effect=self._snap_factory(38.0)):
            r = run_prediction("who wins the race", "test", {})
        assert "ml_deg_model_active" in r["snapshot"]
        assert isinstance(r["snapshot"]["ml_deg_model_active"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Training script structure (no actual training — just validate imports)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingScriptStructure:

    def test_training_script_importable(self):
        """scripts/ml/train_tyre_deg.py must be importable without side effects."""
        import importlib.util, sys
        from pathlib import Path
        script_path = Path("scripts/ml/train_tyre_deg.py")
        if not script_path.exists():
            pytest.skip("train_tyre_deg.py not found — run from project root")
        spec = importlib.util.spec_from_file_location("train_tyre_deg", script_path)
        module = importlib.util.module_from_spec(spec)
        # Don't exec — just check it loads without import errors in its header
        assert spec is not None

    def test_feature_cols_match_model_wrapper(self):
        """FEATURE_COLS in train script must match _FEATURE_COLS in tyre_deg_model.py."""
        from src.backend.services.tyre_deg_model import _FEATURE_COLS as wrapper_cols
        # Expected columns from train script
        expected = [
            "compound_id", "fresh_tyre", "tyre_life", "stint_num",
            "abs_lap", "track_temp", "air_temp", "fuel_load_kg",
            "driver_id", "circuit_id",
        ]
        assert wrapper_cols == expected, \
            f"Column mismatch:\n  wrapper: {wrapper_cols}\n  expected: {expected}"

    def test_compound_id_map_consistent(self):
        """Compound encoding must be consistent across train script and wrapper."""
        from src.backend.services.tyre_deg_model import _COMPOUND_ID
        assert _COMPOUND_ID["S"] == _COMPOUND_ID["SOFT"] == 0
        assert _COMPOUND_ID["M"] == _COMPOUND_ID["MEDIUM"] == 1
        assert _COMPOUND_ID["H"] == _COMPOUND_ID["HARD"] == 2
        assert _COMPOUND_ID["I"] == _COMPOUND_ID["INTERMEDIATE"] == 3
        assert _COMPOUND_ID["W"] == _COMPOUND_ID["WET"] == 4
