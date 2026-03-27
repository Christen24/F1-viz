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

async function fetchJSON<T>(url: string): Promise<T> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status}: ${resp.statusText}`);
    return resp.json();
}

/**
 * Load a session: metadata → events + highlights + insights + laps.
 */
export async function loadSession(year: number, gp: string, session: string) {
    const store = useSessionStore.getState();
    const lapStore = useLapPlaybackStore.getState();

    store.setLoading(true);
    store.setError(null);
    lapStore.reset();

    try {
        // 1. Session metadata
        const meta = await fetchJSON<SessionMeta>(
            `${API}/session?year=${year}&gp=${encodeURIComponent(gp)}&session=${session}`
        );
        store.setMetadata(meta);

        // Auto-select top 3 drivers
        if (meta.drivers?.length) {
            store.setSelectedDrivers(meta.drivers.slice(0, 3).map(d => d.code));
        }

        // 2. Events + highlights + insights + laps + track-replay in parallel
        const sid = meta.session_id;
        const [eventsRes, highlightsRes, insightsRes, lapsRes, trackReplayRes] = await Promise.allSettled([
            fetchJSON<{ events: SessionEvent[] }>(`${API}/session/${sid}/events`),
            fetchJSON<{ highlights: SessionEvent[] }>(`${API}/session/${sid}/highlights`),
            fetchJSON<{ insights: InsightItem[] }>(`${API}/session/${sid}/insights`),
            fetchJSON<LapsResponse>(`${API}/session/${sid}/laps`),
            fetchJSON<TrackReplayResponse>(`${API}/session/${sid}/track-replay`),
        ]);

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
        if (trackReplayRes.status === 'fulfilled') {
            store.setTrackReplay(trackReplayRes.value);
        }

    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        store.setError(msg);
    } finally {
        store.setLoading(false);
    }
}
