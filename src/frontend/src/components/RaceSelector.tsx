/**
 * RaceSelector — Dynamic Year/GP/Session dropdown
 *
 * Fetches the race calendar from FastF1 via /api/calendar/{year}.
 * Automatically shows sprint sessions (Sprint, Sprint Qualifying)
 * only for sprint weekends.
 */
import { useState, useEffect, useCallback } from 'react';
import { loadSession } from '../hooks/useSessionData';
import { useSessionStore } from '../stores/sessionStore';

interface CalendarEvent {
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
    { value: 'FP1', label: 'FP1' },
    { value: 'FP2', label: 'FP2' },
    { value: 'FP3', label: 'FP3' },
];

export function RaceSelector() {
    const [year, setYear] = useState(CURRENT_YEAR);
    const [events, setEvents] = useState<CalendarEvent[]>([]);
    const [selectedGp, setSelectedGp] = useState('');
    const [session, setSession] = useState('R');
    const [calLoading, setCalLoading] = useState(false);
    const loading = useSessionStore(s => s.loading);

    // Fetch calendar when year changes
    const fetchCalendar = useCallback(async (y: number) => {
        setCalLoading(true);
        try {
            const r = await fetch(`/api/calendar/${y}`);
            if (!r.ok) throw new Error('Calendar fetch failed');
            const data = await r.json();
            const evts: CalendarEvent[] = data.events || [];
            setEvents(evts);
            if (evts.length > 0) {
                setSelectedGp(evts[0].event_name);
                setSession('R');
            }
        } catch (err) {
            console.error('Calendar fetch error:', err);
            setEvents([]);
        } finally {
            setCalLoading(false);
        }
    }, []);

    useEffect(() => { fetchCalendar(year); }, [year, fetchCalendar]);

    // Get sessions for selected GP
    const selectedEvent = events.find(e => e.event_name === selectedGp);
    const sessions = selectedEvent?.sessions || FALLBACK_SESSIONS;
    const isSprint = selectedEvent?.is_sprint || false;

    // Reset session to 'R' when GP changes  
    const handleGpChange = (gp: string) => {
        setSelectedGp(gp);
        setSession('R');
    };

    const handleLoad = () => {
        useSessionStore.getState().reset();
        loadSession(year, selectedGp, session);
    };

    return (
        <div className="selector-group" role="search" aria-label="Race selector">
            {/* Year */}
            <select value={year} onChange={e => setYear(Number(e.target.value))} aria-label="Year">
                {YEARS.map(y => <option key={y} value={y}>{y}</option>)}
            </select>

            {/* Grand Prix */}
            <select
                value={selectedGp}
                onChange={e => handleGpChange(e.target.value)}
                disabled={calLoading || events.length === 0}
                aria-label="Grand Prix"
            >
                {calLoading ? (
                    <option>Loading calendar...</option>
                ) : events.length === 0 ? (
                    <option>No events</option>
                ) : (
                    events.map(e => (
                        <option key={e.round} value={e.event_name}>
                            {e.is_sprint ? '⚡ ' : ''}{e.event_name}
                        </option>
                    ))
                )}
            </select>

            {/* Session */}
            <select value={session} onChange={e => setSession(e.target.value)} aria-label="Session type">
                {sessions.map(s => (
                    <option key={s.value} value={s.value}>
                        {s.label}{s.value === 'S' || s.value === 'SQ' ? ' 🏃' : ''}
                    </option>
                ))}
            </select>

            {/* Sprint badge */}
            {isSprint && (
                <span className="sprint-badge">SPRINT WEEKEND</span>
            )}

            {/* Load button */}
            <button
                className="load-btn"
                onClick={handleLoad}
                disabled={loading || calLoading || !selectedGp}
                aria-label="Load race data"
            >
                {loading ? <><span className="spinner-sm" /> Loading…</> : '🏁 Load'}
            </button>
        </div>
    );
}
