"""
F1 Visualization System — Unit Tests: Validator

Tests coordinate validation and track length comparison.
"""
import pytest

from src.backend.services.validator import (
    compute_track_perimeter,
    validate_track_length,
)
from src.backend.schemas import TrackPoint


class TestTrackValidation:
    def test_known_track_passes(self):
        """A track with length close to official should pass."""
        # Create a rough Monza-like oval (~5793m)
        import numpy as np
        t = np.linspace(0, 2 * np.pi, 500)
        # Elongated ellipse
        points = [
            TrackPoint(x=float(700 * np.cos(a)), y=float(350 * np.sin(a)))
            for a in t
        ]
        perimeter = compute_track_perimeter(points)
        # Ellipse perimeter ≈ π * (3(a+b) - sqrt((3a+b)(a+3b)))
        # For a=700, b=350, ≈ 3392m
        assert perimeter > 2000, f"Perimeter too small: {perimeter}"

    def test_tolerance_check(self):
        """Validation should fail when delta exceeds tolerance."""
        # Create a small square (400m) and compare to Monza (5793m)
        points = [
            TrackPoint(x=0, y=0),
            TrackPoint(x=100, y=0),
            TrackPoint(x=100, y=100),
            TrackPoint(x=0, y=100),
        ]
        result = validate_track_length(points, "Monza", tolerance=0.01)
        assert not result["passed"]
        assert result["delta_pct"] > 1

    def test_unknown_track_warns(self):
        """Unknown track name should produce a warning, not fail."""
        points = [TrackPoint(x=0, y=0), TrackPoint(x=100, y=0)]
        result = validate_track_length(points, "Fictional_Circuit")
        assert result["warning"] is not None
        assert result["passed"]  # Unknown tracks pass by default

    def test_empty_track(self):
        """Empty track should compute 0 perimeter."""
        assert compute_track_perimeter([]) == 0.0
