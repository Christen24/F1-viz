/**
 * F1 Visualization System — TypeScript Types
 *
 * All interfaces match the backend JSON schemas.
 */

export interface DriverInfo {
    code: string;
    name: string;
    team: string;
    team_color: string;
    number: number;
}

export interface TrackPoint {
    x: number;
    y: number;
}

export interface SessionMetadata {
    session_id: string;
    year: number;
    gp: string;
    session_type: string;
    circuit_key: number;
    track_name: string;
    track_length_m: number;
    computed_track_length_m: number;
    drivers: DriverInfo[];
    total_laps: number;
    start_time: number;
    end_time: number;
    frame_rate: number;
    chunk_duration_s: number;
    total_frames: number;
    track: TrackPoint[];
    event_count: number;
}

export interface TrackReplayDriver {
    id: string;
    color: string;
}

export interface TrackReplayFrame {
    t: number;
    positions: Record<string, number | {
        d: number;
        s?: number;
        pos?: number;
        x?: number;
        y?: number;
    }>;
}

export interface TrackReplayResponse {
    session_id: string;
    track_points: TrackPoint[];
    track_length: number;
    drivers: TrackReplayDriver[];
    lap_boundaries: number[];
    duration: number;
    frame_rate: number;
    frames: TrackReplayFrame[];
}

/* ===== Lap Playback Types ===== */

export interface LapEvent {
    type: string;           // overtake | pit_stop | fastest_lap | incident
    actor: string;
    victim?: string | null;
    details: Record<string, unknown>;
}

export interface LapSummary {
    lap: number;
    leader: string;
    positions: Record<string, number>;
    lap_times: Record<string, number>;
    gaps: Record<string, number>;
    tyres: Record<string, string>;
    tyre_ages: Record<string, number>;
    pit_stops: LapEvent[];
    events: LapEvent[];
    avg_speed: Record<string, number>;
    max_speed: Record<string, number>;
    throttle_pct: Record<string, number>;
    brake_pct: Record<string, number>;
    drs_pct: Record<string, number>;
    sector_times: Record<string, number[]>;
}

export interface LapsResponse {
    session_id: string;
    total_laps: number;
    laps: LapSummary[];
}

/* ===== Legacy Telemetry Types (kept for compatibility) ===== */

export interface CompactDriverFrame {
    x: number;
    y: number;
    d: number;
    s: number;
    th: number;
    br: number;
    g: number;
    drs: boolean;
    tc: string;
    ta: number;
    l: number;
    syn: boolean;
}

export interface DriverFrame {
    x: number;
    y: number;
    distance: number;
    speed: number;
    throttle: number;
    brake: number;
    gear: number;
    drs: boolean;
    tyreCompound: string;
    tyreAge: number;
    lap: number;
    synthetic: boolean;
    position: number;
}

export interface CompactTimeFrame {
    t: number;
    d: Record<string, CompactDriverFrame>;
}

export interface TimeFrame {
    t: number;
    drivers: Record<string, DriverFrame>;
}

export interface SessionEvent {
    t: number;
    type: string;
    driver: string;
    details: Record<string, unknown>;
    highlightScore: number;
    confidence: number;
    source: string;
}

/** Tire compound colors */
export const COMPOUND_COLORS: Record<string, string> = {
    S: '#ff3333',
    SOFT: '#ff3333',
    M: '#ffcc00',
    MEDIUM: '#ffcc00',
    H: '#ffffff',
    HARD: '#ffffff',
    I: '#44cc44',
    INTERMEDIATE: '#44cc44',
    W: '#3399ff',
    WET: '#3399ff',
    U: '#888888',
    UNKNOWN: '#888888',
};
