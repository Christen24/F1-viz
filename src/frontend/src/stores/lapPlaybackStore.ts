import { create } from 'zustand';
import type { LapSummary } from '../types/session';
import { useSessionStore } from './sessionStore';

interface LapPlaybackState {
    // Data
    lapData: LapSummary[];
    loadingLaps: boolean;

    // Playback
    currentLap: number;
    totalLaps: number;
    isPlaying: boolean;
    speed: number;       // 0.5, 1, 2

    // Continuous Playback Time (Seconds)
    playbackTime: number;

    // Internal
    _lastTick: number | null;
    _rafId: number | null;

    // Actions
    setLapData: (data: LapSummary[]) => void;
    setLoadingLaps: (l: boolean) => void;
    play: () => void;
    pause: () => void;
    togglePlay: () => void;
    nextLap: () => void;
    prevLap: () => void;
    setLap: (n: number) => void;
    setSpeed: (s: number) => void;
    setPlaybackTime: (t: number) => void;
    reset: () => void;
}

function stopTimer(get: () => LapPlaybackState) {
    const id = get()._rafId;
    if (id !== null) {
        cancelAnimationFrame(id);
    }
    get()._lastTick = null;
}

function startTimer(set: any, get: () => LapPlaybackState) {
    stopTimer(get);

    let lastTime = document.timeline ? document.timeline.currentTime as number : performance.now();

    function tick(time: number) {
        const state = get();
        if (!state.isPlaying) return;

        const deltaMs = time - lastTime;
        lastTime = time;

        const sessionStore = useSessionStore.getState();
        const tr = sessionStore.trackReplay;

        if (tr && tr.lap_boundaries && tr.lap_boundaries.length > 0) {
            const maxT = tr.duration;
            const newTime = Math.min(maxT, state.playbackTime + (deltaMs / 1000.0) * state.speed);

            // Find current lap based on lap boundaries
            let newLap = 1;
            const boundaries = tr.lap_boundaries;
            for (let i = 0; i < Math.max(0, boundaries.length - 1); i++) {
                if (newTime >= tr.lap_boundaries[i]) {
                    newLap = i + 1;
                } else {
                    break;
                }
            }
            newLap = Math.max(1, Math.min(newLap, state.totalLaps || 1));

            if (newTime >= maxT) {
                // End of race
                set({ playbackTime: maxT, currentLap: state.totalLaps, isPlaying: false, _rafId: null, _lastTick: null });
                return;
            } else {
                set({ playbackTime: newTime, currentLap: newLap });
            }
        } else {
            // Fallback to legacy discrete lap timer if no track replay data
            // Assumes 1 lap = 20s
            const fallbackMaxT = state.totalLaps * 20;
            const newTime = Math.min(fallbackMaxT, state.playbackTime + (deltaMs / 1000.0) * state.speed);
            const newLap = Math.max(1, Math.min(state.totalLaps, Math.floor(newTime / 20) + 1));

            if (newTime >= fallbackMaxT) {
                set({ playbackTime: fallbackMaxT, currentLap: state.totalLaps, isPlaying: false, _rafId: null, _lastTick: null });
                return;
            } else {
                set({ playbackTime: newTime, currentLap: newLap });
            }
        }

        const rafId = requestAnimationFrame(tick);
        set({ _rafId: rafId, _lastTick: time });
    }

    const rafId = requestAnimationFrame(tick);
    set({ _rafId: rafId, _lastTick: lastTime });
}

export const useLapPlaybackStore = create<LapPlaybackState>((set, get) => ({
    // Data
    lapData: [],
    loadingLaps: false,

    // Playback
    currentLap: 1,
    totalLaps: 0,
    isPlaying: false,
    speed: 1,
    playbackTime: 0,

    // Internal
    _lastTick: null,
    _rafId: null,

    // Actions
    setLapData: (data) => set({
        lapData: data,
        totalLaps: data.length,
        currentLap: 1,
        playbackTime: 0,
        isPlaying: false,
    }),

    setLoadingLaps: (l) => set({ loadingLaps: l }),

    play: () => {
        const s = get();
        if (s.totalLaps === 0) return;

        const sessionStore = useSessionStore.getState();
        const tr = sessionStore.trackReplay;
        const maxT = tr ? tr.duration : (s.totalLaps * 20);

        let startT = s.playbackTime;
        let startL = s.currentLap;

        if (s.playbackTime >= maxT) {
            startT = 0;
            startL = 1;
        }

        set({ isPlaying: true, playbackTime: startT, currentLap: startL });
        startTimer(set, get);
    },

    pause: () => {
        stopTimer(get);
        set({ isPlaying: false, _rafId: null, _lastTick: null });
    },

    togglePlay: () => {
        const s = get();
        if (s.isPlaying) {
            get().pause();
        } else {
            get().play();
        }
    },

    nextLap: () => {
        const s = get();
        if (s.currentLap < s.totalLaps) {
            const nextL = s.currentLap + 1;
            const sessionStore = useSessionStore.getState();
            const tr = sessionStore.trackReplay;
            const newT = (tr && tr.lap_boundaries && tr.lap_boundaries.length > nextL - 1)
                ? tr.lap_boundaries[nextL - 1]
                : (nextL - 1) * 20;

            set({ currentLap: nextL, playbackTime: newT });
        }
    },

    prevLap: () => {
        const s = get();
        if (s.currentLap > 1) {
            const prevL = s.currentLap - 1;
            const sessionStore = useSessionStore.getState();
            const tr = sessionStore.trackReplay;
            const newT = (tr && tr.lap_boundaries && tr.lap_boundaries.length > prevL - 1)
                ? tr.lap_boundaries[prevL - 1]
                : (prevL - 1) * 20;

            set({ currentLap: prevL, playbackTime: newT });
        }
    },

    setLap: (n) => {
        const s = get();
        const clamped = Math.max(1, Math.min(n, s.totalLaps));

        const sessionStore = useSessionStore.getState();
        const tr = sessionStore.trackReplay;
        const newT = (tr && tr.lap_boundaries && tr.lap_boundaries.length > clamped - 1)
            ? tr.lap_boundaries[clamped - 1]
            : (clamped - 1) * 20;

        set({ currentLap: clamped, playbackTime: newT });
    },

    setSpeed: (s) => {
        set({ speed: s });
    },

    setPlaybackTime: (t) => {
        set({ playbackTime: t });
    },

    reset: () => {
        stopTimer(get);
        set({
            lapData: [],
            loadingLaps: false,
            currentLap: 1,
            totalLaps: 0,
            isPlaying: false,
            speed: 1,
            playbackTime: 0,
            _rafId: null,
            _lastTick: null,
        });
    },
}));

/* ===== Selectors ===== */

export function useCurrentLapData(): LapSummary | null {
    const lapData = useLapPlaybackStore((s) => s.lapData);
    const currentLap = useLapPlaybackStore((s) => s.currentLap);
    return lapData[currentLap - 1] ?? null;
}
