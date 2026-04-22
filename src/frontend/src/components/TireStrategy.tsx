/**
 * TireStrategy — 3-Column Bento Stint Bar Component
 */
import { useLapPlaybackStore } from '../stores/lapPlaybackStore';
import { useSessionStore } from '../stores/sessionStore';
import { useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { COMPOUND_COLORS } from '../types/session';

export function TireStrategy() {
    const lapData = useLapPlaybackStore(s => s.lapData);
    const currentLap = useLapPlaybackStore(s => s.currentLap);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const meta = useSessionStore(s => s.metadata);
    const soloDriver = useSessionStore(s => s.soloDriver);

    const [hoverInfo, setHoverInfo] = useState<{
        x: number; y: number; text: string; subtext: string; pitStop: string | null;
    } | null>(null);

    const lapsUpTo = useMemo(() => lapData.slice(0, currentLap), [lapData, currentLap]);

    const stints = useMemo(() => {
        if (!lapsUpTo.length || !meta) {
            return {} as Record<string, { c: string; s: number; e: number; startAge: number; endAge: number; pitDur: string | null }[]>;
        }
        const ds: Record<string, { c: string; s: number; e: number; startAge: number; endAge: number; pitDur: string | null }[]> = {};

        for (const d of meta.drivers) {
            const driverStints: { c: string; s: number; e: number; startAge: number; endAge: number; pitDur: string | null }[] = [];

            for (const lap of lapsUpTo) {
                const compound = lap.tyres[d.code];
                const tyreAge = lap.tyre_ages?.[d.code] ?? 1;
                if (!compound) continue;

                const last = driverStints[driverStints.length - 1];
                if (!last || last.c !== compound || tyreAge < last.endAge) {
                    if (last) {
                        last.e = lap.lap - 1;
                        const prevLapObj = lapsUpTo[lap.lap - 2];
                        if (prevLapObj) {
                            const pit = prevLapObj.pit_stops.find((p: any) => (p.actor || p.driver) === d.code);
                            if (pit && pit.details) {
                                const dur = pit.details.pit_duration || pit.details.duration_s;
                                if (dur) last.pitDur = parseFloat(String(dur)).toFixed(1) + 's';
                            }
                        }
                    }
                    driverStints.push({ c: compound, s: lap.lap, e: lap.lap, startAge: tyreAge, endAge: tyreAge, pitDur: null });
                } else {
                    last.e = lap.lap;
                    last.endAge = tyreAge;
                }
            }
            if (driverStints.length) ds[d.code] = driverStints;
        }
        return ds;
    }, [lapsUpTo, meta]);

    if (!meta) return null;
    const total = totalLaps || meta.total_laps || 50;
    const drivers = meta.drivers.filter(d => stints[d.code]?.length);

    if (!drivers.length) return (
        <div className="tire-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div className="panel-hdr" style={{ padding: '12px 16px', borderBottom: '1px solid var(--glass-border)', fontSize: 11, fontWeight: 800, letterSpacing: 1, color: 'var(--color-text-muted)' }}>
                STINTS
            </div>
            <div className="panel-body" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div className="empty" style={{ padding: 12 }}>
                    <div className="ico">🔄</div>
                    <div style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>Awaiting stint data</div>
                </div>
            </div>
        </div>
    );

    const handleMouseMove = (e: React.MouseEvent, s: any, dCode: string) => {
        const rect = (e.target as HTMLElement).getBoundingClientRect();
        const ageText = s.startAge > 1 ? `${s.startAge} → ${s.endAge}` : s.endAge;
        setHoverInfo({
            x: e.clientX,
            y: rect.top - 8,
            text: `${dCode} • ${s.c} Compound`,
            subtext: `Laps ${s.s}-${s.e} (Tyre Life: ${ageText} laps)`,
            pitStop: s.pitDur ? `Pit ${s.pitDur}` : null
        });
    };

    const handleMouseLeave = () => setHoverInfo(null);
    const displayDrivers = soloDriver ? drivers.filter(d => d.code === soloDriver) : drivers.slice(0, 10);

    return (
        <div className="tire-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
            <div className="panel-hdr" style={{ padding: '12px 16px', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: 1, color: 'var(--color-text-muted)' }}>STINTS</span>
            </div>

            <div className="panel-body" style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
                {displayDrivers.map(d => (
                    <div key={d.code} className={`tire-row-bento ${soloDriver === d.code ? 'solo' : ''}`} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                        <span className="tire-label" style={{ color: d.team_color, fontSize: 13, fontWeight: 800, width: 30 }}>{d.code}</span>
                        <div className="tire-bar" style={{ flex: 1, height: 16, background: 'rgba(255,255,255,0.05)', borderRadius: 4, display: 'flex', overflow: 'hidden' }}>
                            {stints[d.code].map((s, i) => {
                                const age = s.e - s.s + 1;
                                const degradation = Math.max(0.3, 1 - (age * 0.02)); // Subtle visual degradation
                                return (
                                    <div key={i}
                                        className="tire-segment"
                                        onMouseMove={(e) => handleMouseMove(e, s, d.code)}
                                        onMouseLeave={handleMouseLeave}
                                        style={{
                                            width: `${(age / total) * 100}%`,
                                            background: COMPOUND_COLORS[s.c] || '#888',
                                            opacity: degradation,
                                            borderRight: i < stints[d.code].length - 1 ? '2px solid rgba(18, 18, 18, 1)' : 'none',
                                            cursor: 'pointer'
                                        }}
                                    />
                                );
                            })}
                        </div>
                    </div>
                ))}

                <div style={{ display: 'flex', gap: 12, marginTop: 16, fontSize: 10, fontWeight: 700, color: 'var(--color-text-muted)', justifyContent: 'center', fontFamily: 'var(--font-mono)' }}>
                    {['S', 'M', 'H', 'I', 'W'].map(k => (
                        <span key={k} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 2, background: COMPOUND_COLORS[k] }} />
                            {k}
                        </span>
                    ))}
                </div>

                {hoverInfo && createPortal(
                    <div style={{
                        position: 'fixed',
                        left: hoverInfo.x,
                        top: hoverInfo.y,
                        transform: 'translate(-50%, -100%)',
                        background: 'rgba(7, 8, 11, 0.95)',
                        backdropFilter: 'blur(12px)',
                        border: '1px solid rgba(255, 255, 255, 0.15)',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        pointerEvents: 'none',
                        zIndex: 999999,
                        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.8)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '2px',
                        fontFamily: 'var(--font-sans)'
                    }}>
                        <div style={{ fontSize: '11px', fontWeight: 800, color: '#fff' }}>{hoverInfo.text}</div>
                        <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.6)' }}>{hoverInfo.subtext}</div>
                        {hoverInfo.pitStop && (
                            <div style={{ marginTop: '2px', fontSize: '10px', color: 'var(--color-info)', fontWeight: 800 }}>
                                {hoverInfo.pitStop}
                            </div>
                        )}
                    </div>,
                    document.body
                )}
            </div>
        </div>
    );
}
