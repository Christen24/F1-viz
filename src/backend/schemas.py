"""
F1 Visualization System — Pydantic Schemas

All JSON API models. Synthetic fields are explicitly flagged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DriverInfo(BaseModel):
    """Driver metadata."""
    code: str = Field(..., description="Three-letter driver abbreviation")
    name: str = Field(..., description="Full driver name")
    team: str = Field(..., description="Team name")
    team_color: str = Field(..., description="Hex team color, e.g. '#3671C6'")
    number: int = Field(..., description="Car number")


class TrackPoint(BaseModel):
    """Single point on the track centerline, in meters."""
    x: float
    y: float


class VideoSource(BaseModel):
    """Video embed source for a session."""
    provider: str = "youtube"  # youtube | embed
    video_id: str = ""
    fullrace_video_id: str = ""
    embed_url: str = ""
    thumbnail_url: str = ""
    youtube_search_url: str = ""


class SessionMetadata(BaseModel):
    """Lightweight session descriptor returned before any telemetry."""
    session_id: str
    year: int
    gp: str
    session_type: str  # R, Q, FP1, etc.
    track_name: str
    track_length_m: float
    computed_track_length_m: float
    drivers: list[DriverInfo]
    total_laps: int
    start_time: float  # epoch seconds
    end_time: float
    frame_rate: float
    chunk_duration_s: float
    total_frames: int
    video_source: VideoSource | None = None


class DriverFrame(BaseModel):
    """Per-driver data at a single time instant."""
    x: float
    y: float
    distance: float = 0.0
    speed: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    gear: int = 0
    drs: bool = False
    tyre_compound: str = "U"
    tyre_age: int = 0
    lap: int = 0
    position: int = 0
    _synthetic: bool = False


class SessionEvent(BaseModel):
    """Detected event (rule-based or ML-scored)."""
    t: float
    event_type: str  # overtake, pit_stop, fastest_lap, incident
    driver: str
    details: dict = Field(default_factory=dict)
    highlight_score: float = 0.0
    confidence: float = 1.0
    source: str = "rule"  # rule | ml


class TimeFrame(BaseModel):
    """A single time-synchronized frame with all drivers."""
    t: float
    drivers: dict[str, DriverFrame]
    events: list[SessionEvent] = Field(default_factory=list)


class TelemetryChunk(BaseModel):
    """A chunk of telemetry frames for streaming."""
    session_id: str
    start: float
    duration: float
    frame_rate: float
    frames: list[TimeFrame]


class SessionData(BaseModel):
    """Full session data (used internally; API serves metadata + chunks)."""
    metadata: SessionMetadata
    track: list[TrackPoint]
    frames: list[TimeFrame]
    events: list[SessionEvent]


class InsightItem(BaseModel):
    """A single AI-generated insight."""
    category: str  # strategy | tire | overtake | pace | general
    text: str
    confidence: float = 0.8
    related_drivers: list[str] = Field(default_factory=list)
    related_lap: int | None = None


class LapEvent(BaseModel):
    """An event that occurred during a specific lap."""
    type: str  # overtake | pit_stop | fastest_lap | incident
    actor: str
    victim: str | None = None
    details: dict = Field(default_factory=dict)


class LapSummary(BaseModel):
    """Per-lap race summary — drives the Lap Playback Engine."""
    lap: int
    leader: str
    positions: dict[str, int]
    lap_times: dict[str, float]
    gaps: dict[str, float]
    tyres: dict[str, str]
    tyre_ages: dict[str, int]
    pit_stops: list[LapEvent] = Field(default_factory=list)
    events: list[LapEvent] = Field(default_factory=list)
    avg_speed: dict[str, float] = Field(default_factory=dict)
    max_speed: dict[str, float] = Field(default_factory=dict)
    throttle_pct: dict[str, float] = Field(default_factory=dict)
    brake_pct: dict[str, float] = Field(default_factory=dict)
    drs_pct: dict[str, float] = Field(default_factory=dict)
    sector_times: dict[str, list[float]] = Field(default_factory=dict)
