/**
 * Session Store — Zustand
 * Global state for the Race Intelligence Platform.
 */
import { create } from 'zustand';

import type { TrackReplayResponse } from '../types/session';

/* ===== TYPES ===== */
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
export interface VideoSource {
    provider: string;
    video_id: string;
    fullrace_video_id: string;
    embed_url: string;
    thumbnail_url: string;
    youtube_search_url: string;
}
export interface SessionMeta {
    session_id: string;
    year: number;
    gp: string;
    session_type: string;
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
    track?: TrackPoint[];
    video_source?: VideoSource | null;
    event_count?: number;
}
export interface SessionEvent {
    type: string;
    lap: number;
    actor: string;
    victim?: string;
    description?: string;
    highlight_score: number;
    // Legacy fields (from old pipeline)
    t?: number;
    event_type?: string;
    driver?: string;
    details?: Record<string, unknown>;
    confidence?: number;
    source?: string;
}
export interface InsightItem {
    category: string;
    text: string;
    priority?: number;
    confidence?: number;
    related_drivers?: string[];
    related_lap?: number | null;
}
export interface RetiredDriver {
    driver: string;
    name: string;
    team: string;
    retired_lap: number;
    reason: string;
}
export interface TelemetryFrame {
    t: number;
    d: Record<string, {
        x: number; y: number; d: number; s: number;
        th: number; br: number; g: number; drs: boolean;
        tc: string; ta: number; l: number; syn: boolean;
        pos?: number;
    }>;
}

/* ===== STATE ===== */
interface SessionState {
    // Session data
    metadata: SessionMeta | null;
    events: SessionEvent[];
    highlights: SessionEvent[];
    insights: InsightItem[];
    retirements: RetiredDriver[];
    loading: boolean;
    error: string | null;

    // Telemetry
    telemetryFrames: TelemetryFrame[];
    loadingTelemetry: boolean;
    trackReplay: TrackReplayResponse | null;

    // Video sync
    videoReady: boolean;
    videoTime: number;       // current video time in seconds
    videoOffset: number;     // video time offset vs telemetry time
    playState: 'playing' | 'paused' | 'idle';

    // Sync / cursor
    syncTime: number;        // current session master time
    currentLap: number;

    // UI selections
    selectedDrivers: string[];
    soloDriver: string | null;
    focusLap: number | null;
    focusTime: number | null;
    hoverLap: number | null;
    lapRange: [number, number] | null;

    // Actions
    setMetadata: (m: SessionMeta) => void;
    setEvents: (e: SessionEvent[]) => void;
    setHighlights: (h: SessionEvent[]) => void;
    setInsights: (i: InsightItem[]) => void;
    setRetirements: (r: RetiredDriver[]) => void;
    setLoading: (l: boolean) => void;
    setError: (e: string | null) => void;
    setTelemetryFrames: (f: TelemetryFrame[]) => void;
    appendTelemetryFrames: (f: TelemetryFrame[]) => void;
    setLoadingTelemetry: (l: boolean) => void;
    setTrackReplay: (r: TrackReplayResponse | null) => void;
    setVideoReady: (r: boolean) => void;
    setVideoTime: (t: number) => void;
    setVideoOffset: (o: number) => void;
    setPlayState: (s: 'playing' | 'paused' | 'idle') => void;
    setSyncTime: (t: number) => void;
    setCurrentLap: (l: number) => void;
    toggleDriver: (code: string) => void;
    setSelectedDrivers: (codes: string[]) => void;
    setSoloDriver: (code: string | null) => void;
    setFocusLap: (lap: number | null) => void;
    setFocusTime: (t: number | null) => void;
    setHoverLap: (lap: number | null) => void;
    setLapRange: (r: [number, number] | null) => void;
    reset: () => void;
}

const INITIAL: Omit<SessionState, 'setMetadata' | 'setEvents' | 'setHighlights' | 'setInsights' | 'setRetirements' | 'setLoading' | 'setError' | 'setTelemetryFrames' | 'appendTelemetryFrames' | 'setLoadingTelemetry' | 'setTrackReplay' | 'setVideoReady' | 'setVideoTime' | 'setVideoOffset' | 'setPlayState' | 'setSyncTime' | 'setCurrentLap' | 'toggleDriver' | 'setSelectedDrivers' | 'setSoloDriver' | 'setFocusLap' | 'setFocusTime' | 'setHoverLap' | 'setLapRange' | 'reset'> = {
    metadata: null, events: [], highlights: [], insights: [], retirements: [],
    loading: false, error: null, telemetryFrames: [], loadingTelemetry: false,
    trackReplay: null,
    videoReady: false, videoTime: 0, videoOffset: 0, playState: 'idle',
    syncTime: 0, currentLap: 0,
    selectedDrivers: [], soloDriver: null,
    focusLap: null, focusTime: null, hoverLap: null, lapRange: null,
};

export const useSessionStore = create<SessionState>((set, get) => ({
    ...INITIAL,
    setMetadata: (m) => set({ metadata: m }),
    setEvents: (e) => set({ events: e }),
    setHighlights: (h) => set({ highlights: h }),
    setInsights: (i) => set({ insights: i }),
    setRetirements: (r) => set({ retirements: r }),
    setLoading: (l) => set({ loading: l }),
    setError: (e) => set({ error: e }),
    setTelemetryFrames: (f) => set({ telemetryFrames: f }),
    appendTelemetryFrames: (newFrames) => {
        const existing = get().telemetryFrames;
        const merged = [...existing, ...newFrames].sort((a, b) => a.t - b.t);
        const deduped = merged.filter((f, i) => i === 0 || f.t !== merged[i - 1].t);
        set({ telemetryFrames: deduped });
    },
    setLoadingTelemetry: (l) => set({ loadingTelemetry: l }),
    setTrackReplay: (r) => set({ trackReplay: r }),
    setVideoReady: (r) => set({ videoReady: r }),
    setVideoTime: (t) => set({ videoTime: t }),
    setVideoOffset: (o) => set({ videoOffset: o }),
    setPlayState: (s) => set({ playState: s }),
    setSyncTime: (t) => set({ syncTime: t }),
    setCurrentLap: (l) => set({ currentLap: l }),
    toggleDriver: (code) => set((s) => ({
        selectedDrivers: s.selectedDrivers.includes(code)
            ? s.selectedDrivers.filter(c => c !== code)
            : [...s.selectedDrivers, code],
    })),
    setSelectedDrivers: (codes) => set({ selectedDrivers: codes }),
    setSoloDriver: (code) => set({ soloDriver: code }),
    setFocusLap: (lap) => set({ focusLap: lap }),
    setFocusTime: (t) => set({ focusTime: t }),
    setHoverLap: (lap) => set({ hoverLap: lap }),
    setLapRange: (r) => set({ lapRange: r }),
    reset: () => set({ ...INITIAL }),
}));
