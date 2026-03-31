/**
 * useSessionData — API helper + session loader
 *
 * Loads metadata, events, highlights, insights, and lap summaries.
 * Lap data is prefetched in full on session load.
 */
import { useSessionStore } from '../stores/sessionStore';
import { useLapPlaybackStore } from '../stores/lapPlaybackStore';
import type { SessionMeta, SessionEvent, InsightItem } from '../stores/sessionStore';
import type { LapsResponse, TrackReplayResponse } from '../types/session';

const API = '/api';
let activeLoadId = 0;
let activeAbortController: AbortController | null = null;

async function fetchJSON<T>(url: string, signal?: AbortSignal): Promise<T> {
    const resp = await fetch(url, { signal });
    if (!resp.ok) throw new Error(`${resp.status}: ${resp.statusText}`);
    return resp.json();
}

async function fetchTrackReplayWithRetry(
    sessionId: string,
    signal?: AbortSignal,
    attempts = 5,
): Promise<TrackReplayResponse> {
    let lastErr: unknown = null;
    for (let i = 0; i < attempts; i++) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        try {
            return await fetchJSON<TrackReplayResponse>(`${API}/session/${sessionId}/track-replay`, signal);
        } catch (err) {
            if (signal?.aborted) throw err;
            lastErr = err;
            await new Promise((resolve) => setTimeout(resolve, 800 * (i + 1)));
        }
    }
    throw lastErr instanceof Error ? lastErr : new Error('Track replay unavailable');
}

/**
 * Load a session: metadata → events + highlights + insights + laps.
 */
export async function loadSession(year: number, gp: string, session: string) {
    activeLoadId += 1;
    const loadId = activeLoadId;
    activeAbortController?.abort();
    const abortController = new AbortController();
    activeAbortController = abortController;

    const store = useSessionStore.getState();
    const lapStore = useLapPlaybackStore.getState();
    const isStale = () => loadId !== activeLoadId || abortController.signal.aborted;

    store.setLoading(true);
    store.setError(null);
    lapStore.reset();

    try {
        // 1. Session metadata
        const meta = await fetchJSON<SessionMeta>(
            `${API}/session?year=${year}&gp=${encodeURIComponent(gp)}&session=${session}`,
            abortController.signal,
        );
        if (isStale()) return;
        store.setMetadata(meta);

        // Auto-select top 3 drivers
        if (meta.drivers?.length) {
            store.setSelectedDrivers(meta.drivers.slice(0, 3).map(d => d.code));
        }

        // 2. Replay-critical and secondary data loading.
        const sid = meta.session_id;
        const [eventsRes, highlightsRes, insightsRes, lapsRes] = await Promise.allSettled([
            fetchJSON<{ events: SessionEvent[] }>(`${API}/session/${sid}/events`, abortController.signal),
            fetchJSON<{ highlights: SessionEvent[] }>(`${API}/session/${sid}/highlights`, abortController.signal),
            fetchJSON<{ insights: InsightItem[] }>(`${API}/session/${sid}/insights`, abortController.signal),
            fetchJSON<LapsResponse>(`${API}/session/${sid}/laps`, abortController.signal),
        ]);
        if (isStale()) return;

        // Track replay is mandatory for accurate playback/track visuals.
        const replay = await fetchTrackReplayWithRetry(sid, abortController.signal, 5);
        if (isStale()) return;
        store.setTrackReplay(replay);

        if (eventsRes.status === 'fulfilled') {
            const arr = eventsRes.value?.events;
            store.setEvents(Array.isArray(arr) ? arr : []);
        }
        if (highlightsRes.status === 'fulfilled') {
            const arr = highlightsRes.value?.highlights;
            store.setHighlights(Array.isArray(arr) ? arr : []);
        }
        if (insightsRes.status === 'fulfilled') {
            const arr = insightsRes.value?.insights;
            store.setInsights(Array.isArray(arr) ? arr : []);
        }
        if (lapsRes.status === 'fulfilled') {
            const laps = lapsRes.value?.laps;
            if (Array.isArray(laps) && laps.length > 0) {
                lapStore.setLapData(laps);
            }
        }
    } catch (e) {
        if (abortController.signal.aborted) return;
        const msg = e instanceof Error ? e.message : String(e);
        store.setError(msg);
    } finally {
        if (!isStale()) {
            store.setLoading(false);
        }
    }
}
