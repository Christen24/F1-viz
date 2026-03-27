/**
 * HighlightCards - Scrollable list of top highlights
 *
 * Handles both old-format events (t, event_type, driver, details)
 * and new-format events (type, lap, actor, description).
 */
import { useSessionStore } from '../stores/sessionStore';

const ICONS: Record<string, string> = {
    overtake: 'O',
    pit_stop: 'P',
    fastest_lap: 'F',
    incident: '!',
};

const LABELS: Record<string, string> = {
    overtake: 'Overtake',
    pit_stop: 'Pit Stop',
    fastest_lap: 'Fastest Lap',
    incident: 'Incident',
};

export function HighlightCards() {
    const highlights = useSessionStore(s => s.highlights);
    const metadata = useSessionStore(s => s.metadata);

    if (!highlights.length) {
        return (
            <div className="empty" style={{ padding: '16px 0' }}>
                <div style={{ fontSize: 11, opacity: 0.5 }}>No highlights yet</div>
            </div>
        );
    }

    return (
        <div className="hl-cards" role="list" aria-label="Race highlights">
            {highlights.map((hl, i) => {
                // Normalize fields: support both old and new format
                const eventType = hl.type || hl.event_type || 'event';
                const actor = hl.actor || hl.driver || '';
                const victim = hl.victim || (hl.details?.victim as string) || '';
                const lap = hl.lap || (hl.details?.lap_number as number) || 0;
                const score = hl.highlight_score ?? 0;
                const desc = hl.description || '';

                let title = `${actor} — ${LABELS[eventType] || eventType}`;
                if (eventType === 'overtake' && victim) {
                    title = `${actor} overtakes ${victim}`;
                } else if (desc) {
                    title = desc;
                }

                return (
                    <div key={i} className="hl-card" role="listitem">
                        <div className={`hl-badge ${eventType}`}>{ICONS[eventType] || '#'}</div>
                        <div className="hl-body">
                            <div className="hl-title">{title}</div>
                            <div className="hl-meta">{lap ? `Lap ${lap}` : ''}</div>
                        </div>
                        <div className="hl-score">{(score * 100).toFixed(0)}</div>
                    </div>
                );
            })}
        </div>
    );
}
