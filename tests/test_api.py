"""
F1 Visualization System — Unit Tests: API Endpoints

Tests the FastAPI session router using TestClient.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.backend.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestSessionsEndpoint:
    def test_list_sessions_empty(self):
        """Should return empty list when no sessions processed."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data


class TestSessionEndpoint:
    def test_invalid_year(self):
        """Year out of range should return 422."""
        response = client.get("/api/session?year=1990&gp=Monza&session=R")
        assert response.status_code == 422

    def test_invalid_session_type(self):
        """Invalid session type should return 422."""
        response = client.get("/api/session?year=2024&gp=Monza&session=INVALID")
        assert response.status_code == 422


class TestTelemetryEndpoint:
    def test_missing_session(self):
        """Non-existent session should return 404."""
        response = client.get("/api/session/nonexistent_session/telemetry?start=0&duration=30")
        assert response.status_code == 404

    def test_duration_bounds(self):
        """Duration outside bounds should return 422."""
        response = client.get("/api/session/test/telemetry?start=0&duration=0")
        assert response.status_code == 422


class TestEventsEndpoint:
    def test_missing_session(self):
        """Non-existent session should return 404."""
        response = client.get("/api/session/nonexistent/events")
        assert response.status_code == 404


class TestLapsEndpoint:
    def test_missing_session(self):
        """Non-existent session should return 404 or 400."""
        response = client.get("/api/session/nonexistent/laps")
        assert response.status_code in (400, 404)

    def test_invalid_session_id(self):
        """Invalid session ID format should return 400."""
        response = client.get("/api/session/bad/laps")
        assert response.status_code == 400

