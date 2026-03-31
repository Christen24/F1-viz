"""
F1 Visualization System — Unit Tests: Resampler

Tests the master timeline resampler logic.
"""
import numpy as np
import pandas as pd
import pytest

# We test the internal interpolation function directly
from src.backend.services.resampler import (
    _interpolate_driver_to_timeline,
    extract_track_outline,
    compute_track_length,
)
from src.backend.schemas import TrackPoint


class TestInterpolation:
    """Test the driver-to-timeline interpolation logic."""

    def _make_driver_df(self, n_points=100) -> pd.DataFrame:
        """Create a simple driver DataFrame for testing."""
        t = np.linspace(0, 100, n_points)
        df = pd.DataFrame({
            "SessionTime_s": t,
            "X": np.sin(t / 10) * 500,  # circular-ish motion
            "Y": np.cos(t / 10) * 500,
            "Distance": t * 50,  # 50m per second
            "Speed": 200 + np.sin(t) * 50,
            "Throttle": 80 + np.sin(t) * 20,
            "Brake": np.maximum(0, -np.sin(t) * 50),
            "nGear": np.clip(np.round(5 + np.sin(t)), 1, 8).astype(int),
            "DRS": np.zeros(n_points),
            "LapNumber": np.clip(np.floor(t / 80) + 1, 1, 100).astype(int),
            "Compound": "SOFT",
            "TyreLife": np.clip(np.floor(t / 80) * 5, 0, 50).astype(int),
        })
        return df

    def test_interpolation_produces_correct_count(self):
        """Number of output frames should match master timeline length."""
        df = self._make_driver_df(100)
        master_times = np.arange(0, 100, 0.25)  # 4 Hz

        frames = _interpolate_driver_to_timeline(df, master_times)
        assert len(frames) == len(master_times)

    def test_interpolation_output_within_bounds(self):
        """Interpolated values should be within reasonable bounds."""
        df = self._make_driver_df(100)
        master_times = np.arange(0, 100, 0.25)

        frames = _interpolate_driver_to_timeline(df, master_times)

        for frame in frames:
            assert -600 < frame.x < 600, f"X out of bounds: {frame.x}"
            assert -600 < frame.y < 600, f"Y out of bounds: {frame.y}"
            assert 0 <= frame.speed <= 400
            assert 0 <= frame.throttle <= 100
            assert 0 <= frame.brake <= 100
            assert 1 <= frame.gear <= 8

    def test_interpolation_at_original_times(self):
        """At original sample times, values should be close to original."""
        df = self._make_driver_df(100)
        # Use original times as master timeline
        master_times = df["SessionTime_s"].values

        frames = _interpolate_driver_to_timeline(df, master_times)

        # Compare first 10 frames
        for i in range(10):
            assert abs(frames[i].x - df.iloc[i]["X"]) < 1.0  # sub-meter accuracy
            assert abs(frames[i].y - df.iloc[i]["Y"]) < 1.0

    def test_interpolation_monotonic_distance(self):
        """Distance should be mostly monotonically increasing."""
        df = self._make_driver_df(100)
        master_times = np.arange(0, 100, 0.25)

        frames = _interpolate_driver_to_timeline(df, master_times)

        distances = [f.distance for f in frames]
        # Allow some tolerance for interpolation artifacts at boundaries
        strictly_increasing = sum(
            1 for i in range(1, len(distances))
            if distances[i] >= distances[i - 1] - 1  # 1m tolerance
        )
        ratio = strictly_increasing / (len(distances) - 1)
        assert ratio > 0.95, f"Distance not monotonic: {ratio:.2%}"


class TestTrackLength:
    """Test track length computation."""

    def test_square_track(self):
        """A square should have perimeter 4 * side_length."""
        points = [
            TrackPoint(x=0, y=0),
            TrackPoint(x=100, y=0),
            TrackPoint(x=100, y=100),
            TrackPoint(x=0, y=100),
        ]
        length = compute_track_length(points)
        assert abs(length - 400) < 1, f"Expected ~400, got {length}"

    def test_empty_track(self):
        """Empty track should return 0."""
        assert compute_track_length([]) == 0

    def test_single_point(self):
        """Single point should return 0."""
        assert compute_track_length([TrackPoint(x=0, y=0)]) == 0
