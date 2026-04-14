from src.backend import schemas
from src.backend.services import events as events_service
from src.backend.config import settings


def _frame(t: float, a_dist: float, b_dist: float) -> schemas.TimeFrame:
    return schemas.TimeFrame(
        t=t,
        drivers={
            "AAA": schemas.DriverFrame(
                x=0.0,
                y=0.0,
                distance=a_dist,
                speed=305.0,
                throttle=95.0,
                brake=0.0,
                gear=8,
                drs=True,
                lap=5,
            ),
            "BBB": schemas.DriverFrame(
                x=0.0,
                y=0.0,
                distance=b_dist,
                speed=296.0,
                throttle=88.0,
                brake=0.0,
                gear=8,
                drs=False,
                lap=5,
            ),
        },
    )


def test_overtake_detection_rule_only(monkeypatch):
    monkeypatch.setattr(settings, "overtake_ml_enabled", False)

    frames = [
        _frame(0.0, 1000.0, 1020.0),  # AAA behind BBB
        _frame(0.5, 1015.0, 1025.0),  # still behind
        _frame(1.0, 1030.0, 1028.0),  # AAA ahead (swap + proximity)
    ]

    found = events_service._detect_overtakes(frames)
    assert len(found) >= 1
    evt = found[0]
    assert evt.event_type == "overtake"
    assert evt.driver == "AAA"
    assert evt.source == "rule"
    assert "ml_probability" not in evt.details


def test_overtake_detection_hybrid_ml(monkeypatch):
    monkeypatch.setattr(settings, "overtake_ml_enabled", True)
    monkeypatch.setattr(settings, "overtake_ml_blend_weight", 0.6)
    monkeypatch.setattr(settings, "overtake_ml_min_probability", 0.0)
    monkeypatch.setattr(events_service, "predict_overtake_probability", lambda _: 0.9)

    frames = [
        _frame(0.0, 1000.0, 1020.0),
        _frame(0.5, 1015.0, 1025.0),
        _frame(1.0, 1030.0, 1028.0),
    ]

    found = events_service._detect_overtakes(frames)
    assert len(found) >= 1
    evt = found[0]
    assert evt.source == "hybrid_ml"
    assert "ml_probability" in evt.details
    assert 0.0 <= float(evt.details["ml_probability"]) <= 1.0

