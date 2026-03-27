/**
 * Leaderboard — Per-lap positions with FLIP animations & Sparklines (Bento Redesign)
 */
import { useCurrentLapData, useLapPlaybackStore } from '../stores/lapPlaybackStore';
import { useSessionStore } from '../stores/sessionStore';
import { useMemo, useRef, useLayoutEffect, useCallback, useState } from 'react';
import { COMPOUND_COLORS } from '../types/session';

/** 
 * MiniSparkline SVG Component
 * Draws a simple line chart for the last N lap times of a driver.
 */
function MiniSparkline({ data, color }: { data: number[], color: string }) {
    if (data.length < 2) return <svg width="40" height="16" className="sparkline" />;

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const pts = data.map((val, i) => {
        const x = (i / (data.length - 1)) * 40;
        // Invert Y so faster (lower) time is higher on the chart
        const y = 14 - (((val - min) / range) * 12);
        return `${x},${y}`;
    }).join(' ');

    return (
        <svg width="40" height="16" className="sparkline" style={{ overflow: 'visible', opacity: 0.8 }}>
            <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="40" cy={14 - (((data[data.length - 1] - min) / range) * 12)} r="2" fill={color} />
        </svg>
    );
}

export function Leaderboard() {
    const lapData = useCurrentLapData();
    const currentLapIdx = useLapPlaybackStore(s => s.currentLap) - 1;
    const allLaps = useLapPlaybackStore(s => s.lapData);
    const meta = useSessionStore(s => s.metadata);
    const solo = useSessionStore(s => s.soloDriver);
    const [showAll, setShowAll] = useState(false);

    const prevLap = currentLapIdx > 0 ? allLaps[currentLapIdx - 1] : null;

    const entries = useMemo(() => {
        if (!lapData || !meta) return [];
        const list: {
            code: string; color: string; pos: number;
            gap: string; compound: string; posChange: number;
            recentPace: number[];
        }[] = [];

        // Collect last 5 lap times for sparklines
        const startIdx = Math.max(0, currentLapIdx - 4);
        const recentLaps = allLaps.slice(startIdx, currentLapIdx + 1);

        for (const d of meta.drivers) {
            const pos = lapData.positions[d.code];
            if (pos === undefined) continue;

            const gap = lapData.gaps[d.code] ?? 0;
            const compound = lapData.tyres[d.code] ?? 'U';
            const prevPos = prevLap?.positions[d.code] ?? pos;
            const posChange = prevPos - pos;

            const recentPace = recentLaps.map(l => l.lap_times?.[d.code] || 0).filter(t => t > 0);

            list.push({
                code: d.code,
                color: d.team_color,
                pos,
                gap: pos === 1 ? 'LDR' : `+${gap.toFixed(1)}`,
                compound,
                posChange,
                recentPace
            });
        }
        list.sort((a, b) => a.pos - b.pos);
        return list;
    }, [lapData, prevLap, meta, allLaps, currentLapIdx]);

    const displayEntries = showAll ? entries : entries.slice(0, 8);

    // FLIP refs
    const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());
    const prevPositions = useRef<Map<string, DOMRect>>(new Map());

    useLayoutEffect(() => {
        const nextPositions = new Map<string, DOMRect>();
        rowRefs.current.forEach((el, code) => {
            nextPositions.set(code, el.getBoundingClientRect());
        });

        prevPositions.current.forEach((prevRect, code) => {
            const el = rowRefs.current.get(code);
            const nextRect = nextPositions.get(code);
            if (!el || !nextRect) return;

            const deltaY = prevRect.top - nextRect.top;
            if (Math.abs(deltaY) > 1) {
                el.style.transform = `translateY(${deltaY}px)`;
                el.style.transition = 'none';
                requestAnimationFrame(() => {
                    el.style.transform = '';
                    el.style.transition = 'transform 0.4s cubic-bezier(0.22, 1, 0.36, 1)';
                });
            }
        });

        prevPositions.current = nextPositions;
    }, [displayEntries]);

    const setRef = useCallback((code: string) => (el: HTMLDivElement | null) => {
        if (el) rowRefs.current.set(code, el);
        else rowRefs.current.delete(code);
    }, []);

    if (!meta || !lapData) return null;

    const handleClick = (code: string, e: React.MouseEvent) => {
        if (e.shiftKey) {
            useSessionStore.getState().setSoloDriver(solo === code ? null : code);
        } else {
            useSessionStore.getState().toggleDriver(code);
        }
    };

    return (
        <div className="lb-wrap" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div className="lb-hdr">
                <span>CLASSIFICATION</span>
                <span className="lb-hdr-right">PACE (5L)</span>
            </div>

            <div className="lb-body" style={{ flex: 1, overflowY: 'auto' }}>
                {displayEntries.map((e) => (
                    <div
                        key={e.code}
                        ref={setRef(e.code)}
                        className={`lb-row-bento ${solo === e.code ? 'solo' : ''}`}
                        onClick={(ev) => handleClick(e.code, ev)}
                        role="button"
                    >
                        <div className="lb-left">
                            <span className="lb-pos-num">{e.pos}</span>
                            <span className="lb-team-bar" style={{ background: e.color }} />
                            <span className="lb-driver-code">{e.code}</span>

                            {e.posChange !== 0 ? (
                                <span className={`lb-pos-change ${e.posChange > 0 ? 'up' : 'down'}`}>
                                    {e.posChange > 0 ? `▲${e.posChange}` : `▼${Math.abs(e.posChange)}`}
                                </span>
                            ) : <span className="lb-pos-change-empty" />}
                        </div>

                        <div className="lb-right">
                            <span className="lb-gap-mono">{e.gap}</span>
                            <div className="lb-spark-container">
                                <MiniSparkline data={e.recentPace} color={e.color} />
                            </div>
                            <span
                                className="lb-tire-pill"
                                style={{ background: COMPOUND_COLORS[e.compound] || '#888' }}
                            >
                                {e.compound}
                            </span>
                        </div>
                    </div>
                ))}

                {entries.length > 8 && (
                    <button className="lb-collapse-btn" onClick={() => setShowAll(s => !s)}>
                        {showAll ? 'Show Top 8' : `Show All (${entries.length})`}
                    </button>
                )}
            </div>
        </div>
    );
}
