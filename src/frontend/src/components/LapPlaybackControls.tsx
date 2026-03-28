/**
 * LapPlaybackControls — Scrub-Sync Heatmap Timeline (2026 Redesign)
 * 
 * Integrated seamlessly into the base of the Video Stage.
 * Features a heatmap timeline (brighter areas for overtakes/action) 
 * and subtle vertical "Glimmer" lines for Pit Stops and Fastest Laps.
 */
import { useLapPlaybackStore } from '../stores/lapPlaybackStore';
import { useSessionStore } from '../stores/sessionStore';
import { useMemo, useState, useRef } from 'react';

const SPEEDS = [0.5, 1, 2];

const formatTime = (seconds: number) => {
    const safe = Math.max(0, Math.floor(seconds || 0));
    const mins = Math.floor(safe / 60);
    const secs = safe % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
};

export function LapPlaybackControls() {
    const currentLap = useLapPlaybackStore(s => s.currentLap);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const isPlaying = useLapPlaybackStore(s => s.isPlaying);
    const speed = useLapPlaybackStore(s => s.speed);
    const lapData = useLapPlaybackStore(s => s.lapData);
    const loading = useLapPlaybackStore(s => s.loadingLaps);

    const [hoverLap, setHoverLap] = useState<number | null>(null);
    const [hoverTime, setHoverTime] = useState<number | null>(null);
    const [hoverPercent, setHoverPercent] = useState<number | null>(null);
    const [scrubbing, setScrubbing] = useState(false);
    const trackRef = useRef<HTMLDivElement>(null);

    // Collect pit stops and overtakes for the heatmap & glimmers
    const markers = useMemo(() => {
        const pits: { lap: number; driver: string; detailStr: string }[] = [];
        const fastest: { lap: number; driver: string }[] = [];
        const heat: Record<number, number> = {}; // lap -> intensity (0-1)

        let overallFastestLapTime = Infinity;
        let overallFastestLapObj: any = null;

        for (const lap of lapData) {
            let intensity = 0.1; // Base low intensity

            // Pit stops (Glimmer)
            if (lap.pit_stops.length > 0) {
                const p = lap.pit_stops[0] as any;
                const driver = p.actor || p.driver || 'Driver';
                const d = p.details || {};
                let detailStr = '';
                const dur = d.pit_duration ?? d.duration_s;
                if (dur !== undefined && dur !== null) detailStr += `${parseFloat(dur).toFixed(1)}s`;
                if (d.compound) detailStr += detailStr ? ` • ${d.compound}` : d.compound;
                pits.push({ lap: lap.lap, driver, detailStr });
                intensity += 0.4;
            }

            // Overtakes increase heat
            if (lap.events.length > 0) {
                const ots = lap.events.filter(e => e.type === 'overtake').length;
                intensity += Math.min(0.5, ots * 0.15);
            }

            // Find fastest lap times to mark the absolute fastest later
            const lapTimes = lap.lap_times ?? {};
            for (const driver in lapTimes) {
                if (lapTimes[driver] < overallFastestLapTime) {
                    overallFastestLapTime = lapTimes[driver];
                    overallFastestLapObj = { lap: lap.lap, driver };
                }
            }

            heat[lap.lap] = Math.min(1, intensity);
        }

        if (overallFastestLapObj) {
            fastest.push(overallFastestLapObj);
            heat[overallFastestLapObj.lap] = 1.0;
        }

        return { pits, fastest, heat };
    }, [lapData]);

    if (totalLaps === 0) return null;

    const playbackTime = useLapPlaybackStore(s => s.playbackTime);
    const store = useLapPlaybackStore.getState();
    const sessionStore = useSessionStore.getState();
    const tr = sessionStore.trackReplay;
    const maxT = tr ? tr.duration : (Math.max(1, totalLaps) * 20);
    const progressPercent = maxT > 0 ? (playbackTime / maxT) * 100 : 0;
    const bufferedPercent = tr ? 100 : Math.min(100, progressPercent + 12);

    const calculateTimeFromEvent = (e: React.MouseEvent<HTMLDivElement> | MouseEvent) => {
        if (!trackRef.current) return playbackTime;
        const rect = trackRef.current.getBoundingClientRect();
        const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        return percent * maxT;
    };

    const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
        store.setPlaybackTime(calculateTimeFromEvent(e));
    };

    const handleTrackHover = (e: React.MouseEvent<HTMLDivElement>) => {
        if (trackRef.current) {
            const rect = trackRef.current.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            setHoverPercent(pct * 100);
        }
        const timeAtHover = calculateTimeFromEvent(e);
        setHoverTime(timeAtHover);
        let hLap = 1;
        if (tr && tr.lap_boundaries) {
            for (let i = 0; i < Math.max(0, tr.lap_boundaries.length - 1); i++) {
                if (timeAtHover >= tr.lap_boundaries[i]) hLap = i + 1;
                else break;
            }
            hLap = Math.max(1, Math.min(hLap, totalLaps));
        } else {
            hLap = Math.max(1, Math.min(totalLaps, Math.floor(timeAtHover / 20) + 1));
        }
        setHoverLap(hLap);

        if (scrubbing) {
            store.setPlaybackTime(timeAtHover);
        }
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
        setScrubbing(true);
        handleTrackClick(e);
        const handleMouseUp = () => setScrubbing(false);
        const handleMouseMove = (e: MouseEvent) => {
            if (scrubbing) store.setPlaybackTime(calculateTimeFromEvent(e));
        };
        window.addEventListener('mouseup', handleMouseUp, { once: true });
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', () => window.removeEventListener('mousemove', handleMouseMove), { once: true });
    };

    // Calculate dynamic heatmap gradient for the track background
    const heatmapBackground = useMemo(() => {
        if (!totalLaps) return 'rgba(255,255,255,0.05)';
        const stops = [];
        for (let i = 1; i <= totalLaps; i++) {
            const intensity = markers.heat[i] || 0.1;
            // Map intensity to a bright accent color (F1 red variant), or fade back to dark gray
            const alpha = 0.05 + (intensity * 0.4);
            const pct = ((i - 1) / totalLaps) * 100;
            const stopPct = ((i) / totalLaps) * 100;
            // Use blocky banding for heatmap
            stops.push(`rgba(225, 6, 0, ${alpha}) ${pct}%`);
            stops.push(`rgba(225, 6, 0, ${alpha}) ${stopPct}%`);
        }
        return `linear-gradient(to right, ${stops.join(', ')})`;
    }, [totalLaps, markers.heat]);

    return (
        <div className="scrub-sync-container" role="region" aria-label="Race timeline">
            {/* Top row: Transport + Lap Counter + Controls */}
            <div className="scrub-toolbar">
                <div className="scrub-transport">
                    <button className="scrub-btn" onClick={() => store.prevLap()} disabled={currentLap <= 1} title="Previous Lap">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" /></svg>
                    </button>

                    <button className="scrub-btn play-pause" onClick={() => store.togglePlay()} title={isPlaying ? "Pause" : "Play"}>
                        {isPlaying ? (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                        )}
                    </button>

                    <button className="scrub-btn" onClick={() => store.nextLap()} disabled={currentLap >= totalLaps} title="Next Lap">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" /></svg>
                    </button>
                </div>

                {/* Big Lap Counter */}
                <div className="scrub-lap-display">
                    <span className="lap-lbl">LAP</span>
                    <span className="lap-cur">{String(currentLap).padStart(2, '0')}</span>
                    <span className="lap-div">/</span>
                    <span className="lap-tot">{totalLaps}</span>
                </div>

                {/* Speed Controls */}
                <div className="scrub-speeds">
                    {SPEEDS.map(s => (
                        <button key={s} className={`speed-btn ${speed === s ? 'active' : ''}`} onClick={() => store.setSpeed(s)}>
                            {s}x
                        </button>
                    ))}
                </div>
            </div>

            {/* Heatmap Timeline */}
            <div className="scrub-timeline"
                ref={trackRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleTrackHover}
                onMouseLeave={() => { setHoverLap(null); setHoverTime(null); setHoverPercent(null); setScrubbing(false); }}
            >
                {/* Background Heatmap */}
                <div className="timeline-bg" style={{ background: heatmapBackground }}></div>

                {/* Buffered Fill */}
                <div className="timeline-buffered" style={{ width: `${bufferedPercent}%` }}></div>

                {/* Progress Fill */}
                <div className="timeline-fill" style={{ width: `${progressPercent}%` }}></div>

                {/* Glimmer Markers */}
                {markers.pits.map((m, i) => (
                    <div
                        key={`pit-${m.lap}-${i}`}
                        className="glimmer glimmer-pit"
                        style={{ left: `${((m.lap - 1) / Math.max(totalLaps - 1, 1)) * 100}%` }}
                        title={`Lap ${m.lap}: Pit Stop (${m.driver})${m.detailStr ? ' - ' + m.detailStr : ''}`}
                    />
                ))}
                {markers.fastest.map((m, i) => (
                    <div
                        key={`fl-${m.lap}-${i}`}
                        className="glimmer glimmer-fastest"
                        style={{ left: `${((m.lap - 1) / Math.max(totalLaps - 1, 1)) * 100}%` }}
                        title={`Lap ${m.lap}: Fastest Lap (${m.driver})`}
                    />
                ))}

                {/* Current Playhead */}
                <div className="timeline-playhead" style={{ left: `${progressPercent}%` }}>
                    <span className={`timeline-scrubber ${scrubbing ? 'active' : ''}`} />
                </div>

                {/* Hover preview line + tooltip */}
                {hoverPercent !== null && (
                    <div className="timeline-preview" style={{ left: `${hoverPercent}%` }}>
                        <span className="timeline-preview-line" />
                        {hoverTime !== null && (
                            <span className="timeline-preview-time">{formatTime(hoverTime)}</span>
                        )}
                    </div>
                )}

                {/* Hover indicator */}
                {hoverLap !== null && hoverLap !== currentLap && (
                    <div className="timeline-hover-tip" style={{ left: `${((hoverLap - 1) / Math.max(totalLaps - 1, 1)) * 100}%` }}>
                        Lap {hoverLap}
                    </div>
                )}
            </div>

            <div className="timeline-time-row" aria-hidden="true">
                <span className="timeline-time-current">{formatTime(playbackTime)}</span>
                <span className="timeline-time-total">{formatTime(maxT)}</span>
            </div>

            {loading && <div className="timeline-loading">Syncing laps...</div>}
        </div>
    );
}
