/**
 * Header — Command Hub (2026 Bento Redesign)
 *
 * Minimalist translucent bar.
 * Replaces bulky "Load" button with auto-fetching.
 * Dropdowns designed to feel like Cmd+K palettes.
 */
import { useState, useEffect, useCallback } from 'react';
import { loadSession } from '../hooks/useSessionData';
import { useSessionStore } from '../stores/sessionStore';

interface CalEvent {
    round: number;
    event_name: string;
    country: string;
    is_sprint: boolean;
    sessions: { value: string; label: string }[];
}

const CURRENT_YEAR = new Date().getFullYear();
const YEARS = Array.from({ length: CURRENT_YEAR - 2018 + 1 }, (_, i) => CURRENT_YEAR - i);

const FALLBACK_SESSIONS = [
    { value: 'R', label: 'Race' },
    { value: 'Q', label: 'Qualifying' },
];

export function Header() {
    const searchParams = new URLSearchParams(window.location.search);
    const initialYear = searchParams.get('year') ? parseInt(searchParams.get('year')!) : Math.min(CURRENT_YEAR, 2024);
    const initialGp = searchParams.get('event') || '';
    const initialSession = searchParams.get('session') === 'Qualifying' ? 'Q' : 'R';

    const [year, setYear] = useState(initialYear);
    const [events, setEvents] = useState<CalEvent[]>([]);
    const [gp, setGp] = useState(initialGp);
    const [session, setSession] = useState(initialSession);
    const [calLoading, setCalLoading] = useState(false);

    const loading = useSessionStore(s => s.loading);

    // Fetch calendar when year changes
    const fetchCalendar = useCallback(async (y: number) => {
        setCalLoading(true);
        try {
            const r = await fetch(`/api/calendar/${y}`);
            if (!r.ok) throw new Error(`Calendar ${y} failed`);
            const data = await r.json();
            const evts: CalEvent[] = data.events || [];
            setEvents(evts);
            if (evts.length > 0) {
                setGp(prevGp => {
                    if (!prevGp) return evts[0].event_name;
                    // If the current gp doesn't exist in this new year's calendar, fall back
                    const exists = evts.find(e => e.event_name === prevGp);
                    return exists ? prevGp : evts[0].event_name;
                });
                setSession(prev => prev || 'R');
            }
        } catch (err) {
            console.error('Calendar fetch error:', err);
            setEvents([]);
        } finally {
            setCalLoading(false);
        }
    }, []);

    useEffect(() => { fetchCalendar(year); }, [year, fetchCalendar]);

    // Auto-fetch session when selections change
    useEffect(() => {
        if (!gp || !session) return;
        useSessionStore.getState().reset();
        loadSession(year, gp, session);
    }, [year, gp, session]);

    const selectedEvent = events.find(e => e.event_name === gp);
    const sessions = selectedEvent?.sessions || FALLBACK_SESSIONS;
    const isSprint = selectedEvent?.is_sprint || false;

    return (
        <header className="cmd-hub">
            <div className="cmd-brand">
                <span className="f1">F1</span>
                <span className="rest">VIZ</span>
            </div>

            <div className="cmd-palette-wrapper">
                {/* Year Select */}
                <div className="cmd-select-group">
                    <span className="cmd-icon">📅</span>
                    <select value={year} onChange={e => setYear(+e.target.value)} className="cmd-input">
                        {YEARS.map(y => <option key={y}>{y}</option>)}
                    </select>
                </div>

                <div className="cmd-divider">/</div>

                {/* GP Select */}
                <div className="cmd-select-group" style={{ flex: 1, minWidth: 200 }}>
                    <span className="cmd-icon">🌍</span>
                    <select
                        value={gp}
                        onChange={e => { setGp(e.target.value); setSession('R'); }}
                        disabled={calLoading || events.length === 0}
                        className="cmd-input"
                    >
                        {calLoading ? <option>Syncing Calendar…</option> :
                            events.length === 0 ? <option>No events</option> :
                                events.map(e => <option key={e.round} value={e.event_name}>{e.is_sprint ? '⚡ ' : ''}{e.event_name}</option>)
                        }
                    </select>
                </div>

                <div className="cmd-divider">/</div>

                {/* Session Select */}
                <div className="cmd-select-group">
                    <span className="cmd-icon">⏱</span>
                    <select value={session} onChange={e => setSession(e.target.value)} className="cmd-input">
                        {sessions.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                    </select>
                </div>
            </div>

            <div className="cmd-status">
                {loading ? (
                    <div className="status-syncing">
                        <span className="spin-small"></span>
                        Syncing Telemetry...
                    </div>
                ) : (
                    <div className="status-ready">
                        <span className="indicator-dot green"></span>
                        Live
                        {isSprint && <span className="sprint-badge" style={{ marginLeft: 8 }}>SPRINT</span>}
                    </div>
                )}
            </div>
        </header>
    );
}
