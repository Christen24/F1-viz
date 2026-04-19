import json

def _gap_trend(laps: list[dict], driver: str, lookback: int = 3) -> float | None:
    """Returns gap change over last N laps. Negative = closing."""
    if not laps:
        return None
    vals = [
        lap["gaps"].get(driver)
        for lap in laps[-lookback:]
        if isinstance((lap.get("gaps") or {}).get(driver), (int, float))
    ]
    if len(vals) < 2:
        return None
    return round(float(vals[-1]) - float(vals[0]), 3)

# Test cases
laps_test = [
    {"gaps": {"LEC": 10.0, "VER": 0.0}},
    {"gaps": {"LEC": 9.5, "VER": 0.0}},
    {"gaps": {"LEC": 9.2, "VER": 0.0}},
]

print(f"Trend (closing): {_gap_trend(laps_test, 'LEC')}") # Expected: 9.2 - 10.0 = -0.8

laps_test_far = [
    {"gaps": {"LEC": 10.0, "VER": 0.0}},
    {"gaps": {"LEC": 10.5, "VER": 0.0}},
    {"gaps": {"LEC": 11.2, "VER": 0.0}},
]
print(f"Trend (losing): {_gap_trend(laps_test_far, 'LEC')}") # Expected: 11.2 - 10.0 = 1.2
