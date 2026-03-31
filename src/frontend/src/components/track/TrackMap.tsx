import { useEffect, useMemo, useState } from 'react';
import { useSessionStore } from '../../stores/sessionStore';
import { useLapPlaybackStore } from '../../stores/lapPlaybackStore';
import type { TrackReplayFrame } from '../../types/session';
import { normalizeTrack, buildTrackPath, upperBound } from './trackUtils';
import { CarsLayer } from './CarsLayer';
import { useRaceAnimation } from './useRaceAnimation';

function getDriverSnapshot(
    frames: TrackReplayFrame[],
    playbackTime: number,
    driverId: string | null,
) {
    if (!driverId || frames.length === 0) return null;
    const frameTimes = frames.map((f) => f.t);
    const hi = upperBound(frameTimes, playbackTime);
    const i0 = Math.max(0, hi - 1);
    const i1 = Math.min(frames.length - 1, hi);
    const f0 = frames[i0];
    const f1 = frames[i1];
    const p0 = f0.positions[driverId];
    const p1 = f1.positions[driverId];

    if (!p0 || !p1) return null;

    const d0 = typeof p0 === 'number' ? p0 : (typeof p0.d === 'number' ? p0.d : null);
    const d1 = typeof p1 === 'number' ? p1 : (typeof p1.d === 'number' ? p1.d : null);
    if (d0 === null || d1 === null) return null;

    const dt = Math.max(f1.t - f0.t, 0.001);
    const alpha = Math.min(1, Math.max(0, (playbackTime - f0.t) / dt));
    const speed0 = (typeof p0 === 'object' && typeof p0.s === 'number') ? p0.s : 0;
    const speed1 = (typeof p1 === 'object' && typeof p1.s === 'number') ? p1.s : speed0;
    const speed = speed0 + (speed1 - speed0) * alpha;

    let pos: number | null = null;
    if (typeof p1 === 'object' && typeof p1.pos === 'number' && p1.pos > 0) pos = p1.pos;
    else if (typeof p0 === 'object' && typeof p0.pos === 'number' && p0.pos > 0) pos = p0.pos;

    // Fallback rank by distance when explicit race position is unavailable.
    if (pos === null) {
        const distances: number[] = [];
        for (const value of Object.values(f1.positions)) {
            if (typeof value === 'number') distances.push(value);
            else if (value && typeof value.d === 'number') distances.push(value.d);
        }
        distances.sort((a, b) => b - a);
        const currentD = d1;
        const rankIndex = distances.findIndex((v) => currentD >= v - 1e-6);
        if (rankIndex >= 0) pos = rankIndex + 1;
    }

    return {
        speed,
        pos,
    };
}

export function TrackMap() {
    const metadata = useSessionStore((s) => s.metadata);
    const trackReplay = useSessionStore((s) => s.trackReplay);
    const currentLap = useLapPlaybackStore((s) => s.currentLap);
    const totalLaps = useLapPlaybackStore((s) => s.totalLaps);
    const playbackTime = useLapPlaybackStore((s) => s.playbackTime);
    const lapData = useLapPlaybackStore((s) => s.lapData);
    const leaderCode = lapData[currentLap - 1]?.leader ?? '--';
    const [focusedDriver, setFocusedDriver] = useState<string | null>(null);

    // Playback time is maintained in Zustand and mirrored into a ref.
    const { sessionTimeRef } = useRaceAnimation();
    const replay = trackReplay;

    // Use the same track source that replay distances are based on.
    const trackPoints = useMemo(
        () => replay?.track_points ?? metadata?.track ?? [],
        [metadata?.track, replay?.track_points],
    );

    const normalized = useMemo(() => normalizeTrack(trackPoints), [trackPoints]);
    const pathD = useMemo(() => buildTrackPath(normalized.points), [normalized.points]);

    const hasCars = Boolean(
        replay
        && replay.frames.length > 0
        && normalized.points.length > 1,
    );

    useEffect(() => {
        if (!replay?.drivers?.length) {
            setFocusedDriver(null);
            return;
        }
        if (focusedDriver && replay.drivers.some((d) => d.id === focusedDriver)) return;
        const defaultDriver = replay.drivers.find((d) => d.id === leaderCode)?.id ?? replay.drivers[0]?.id ?? null;
        setFocusedDriver(defaultDriver);
    }, [focusedDriver, leaderCode, replay]);

    const driverSnapshot = useMemo(
        () => getDriverSnapshot(replay?.frames ?? [], playbackTime, focusedDriver),
        [focusedDriver, playbackTime, replay?.frames],
    );

    return (
        <div className="track-container">
            <div className="track-header">
                <span>LIVE CIRCUIT</span>
                <span className="live-dot">LIVE</span>
            </div>

            <div className="track-title">
                {metadata?.track_name || metadata?.gp || 'Grand Prix'}
            </div>

            {replay && replay.drivers.length > 0 && (
                <div className="track-driver-chips">
                    {replay.drivers.map((d) => (
                        <button
                            key={d.id}
                            className={`track-driver-chip${focusedDriver === d.id ? ' active' : ''}`}
                            style={{
                                borderColor: focusedDriver === d.id ? d.color : undefined,
                                color: focusedDriver === d.id ? '#f5f7ff' : undefined,
                                boxShadow: focusedDriver === d.id ? `inset 0 0 0 1px ${d.color}` : undefined,
                            }}
                            onClick={() => setFocusedDriver(d.id)}
                            type="button"
                        >
                            {d.id}
                        </button>
                    ))}
                </div>
            )}

            <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg
                    viewBox={`0 0 ${normalized.width} ${normalized.height}`}
                    style={{ width: '100%', height: '100%', overflow: 'visible' }}
                >
                    {pathD && (
                        <path
                            d={pathD}
                            stroke="#374151"
                            strokeWidth="8"
                            fill="none"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                        />
                    )}

                    {hasCars && replay && (
                        <CarsLayer
                            drivers={replay.drivers}
                            frames={replay.frames}
                            sessionTimeRef={sessionTimeRef}
                            rawTrackPoints={trackPoints}
                            normalizedTrackPoints={normalized.points}
                            trackLength={replay.track_length}
                            leaderId={leaderCode}
                        />
                    )}
                </svg>

                {!hasCars && (
                    <div style={{
                        position: 'absolute',
                        bottom: 8,
                        left: 8,
                        fontSize: 11,
                        color: '#9ca3af',
                        background: 'rgba(0,0,0,0.45)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: 6,
                        padding: '4px 8px',
                    }}>
                        Waiting for replay frames...
                    </div>
                )}

            </div>

            {focusedDriver && driverSnapshot && (
                <div className="track-driver-insight-dock">
                    <div className="track-driver-insight">
                        <div className="track-driver-insight-title">{focusedDriver} Live Telemetry</div>
                        <div className="track-driver-insight-grid">
                            <div className="track-driver-insight-cell">
                                <div className="track-driver-insight-val">{Math.max(0, driverSnapshot.speed).toFixed(0)} km/h</div>
                                <div className="track-driver-insight-lbl">Speed</div>
                            </div>
                            <div className="track-driver-insight-cell">
                                <div className="track-driver-insight-val">
                                    {driverSnapshot.pos ? `P${driverSnapshot.pos}` : '--'}
                                </div>
                                <div className="track-driver-insight-lbl">Position</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="track-stats">
                <div>Lap {currentLap} / {totalLaps || '--'}</div>
                <div>Ldr: {leaderCode}</div>
            </div>
        </div>
    );
}
