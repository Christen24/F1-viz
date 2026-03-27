/**
 * TelemetrySnapshot — Per-lap telemetry aggregates
 *
 * Shows: avg speed, max speed, throttle %, brake %, tire age, DRS usage %
 */
import { useCurrentLapData } from '../stores/lapPlaybackStore';
import { useSessionStore } from '../stores/sessionStore';
import { useMemo } from 'react';
import { COMPOUND_COLORS } from '../types/session';

export function TelemetrySnapshot() {
    const lapData = useCurrentLapData();
    const soloDriver = useSessionStore(s => s.soloDriver);
    const selectedDrivers = useSessionStore(s => s.selectedDrivers);
    const metadata = useSessionStore(s => s.metadata);

    const snapshot = useMemo(() => {
        if (!lapData || !metadata) return null;
        const code = soloDriver || selectedDrivers[0] || lapData.leader;
        if (!code) return null;

        return {
            code,
            avgSpeed: lapData.avg_speed[code] ?? 0,
            maxSpeed: lapData.max_speed[code] ?? 0,
            throttle: lapData.throttle_pct[code] ?? 0,
            brake: lapData.brake_pct[code] ?? 0,
            tyreAge: lapData.tyre_ages[code] ?? 0,
            compound: lapData.tyres[code] ?? 'U',
            drs: lapData.drs_pct[code] ?? 0,
            sectors: lapData.sector_times[code] ?? [],
        };
    }, [lapData, soloDriver, selectedDrivers, metadata]);

    if (!snapshot || !metadata) {
        return (
            <div className="card">
                <div className="panel-hdr">
                    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="panel-icon">⏱️</span>
                        Telemetry
                    </span>
                </div>
                <div className="panel-body">
                    <div className="empty" style={{ padding: 12 }}>
                        <div style={{ fontSize: 11 }}>Start playback to see lap data</div>
                    </div>
                </div>
            </div>
        );
    }

    const driverColor = metadata.drivers.find(d => d.code === snapshot.code)?.team_color || '#888';

    return (
        <div className="card">
            <div className="panel-hdr">
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="panel-icon">⏱️</span>
                    Telemetry — <span style={{ color: driverColor, marginLeft: 2 }}>{snapshot.code}</span>
                </span>
            </div>
            <div className="panel-body">
                <div className="telem-grid">
                    <div className="telem-cell">
                        <div className="telem-val">{snapshot.avgSpeed.toFixed(0)}</div>
                        <div className="telem-lbl">Avg km/h</div>
                    </div>
                    <div className="telem-cell">
                        <div className="telem-val">{snapshot.maxSpeed.toFixed(0)}</div>
                        <div className="telem-lbl">Max km/h</div>
                    </div>
                    <div className="telem-cell">
                        <div className="telem-val" style={{ color: snapshot.throttle > 50 ? '#00d27a' : '#9898b0' }}>
                            {snapshot.throttle.toFixed(0)}%
                        </div>
                        <div className="telem-lbl">Throttle</div>
                    </div>
                    <div className="telem-cell">
                        <div className="telem-val" style={{ color: snapshot.brake > 10 ? '#e10600' : '#9898b0' }}>
                            {snapshot.brake.toFixed(0)}%
                        </div>
                        <div className="telem-lbl">Brake</div>
                    </div>
                    <div className="telem-cell">
                        <div className="telem-val">{snapshot.tyreAge}</div>
                        <div className="telem-lbl">Tyre Age</div>
                    </div>
                    <div className="telem-cell">
                        <div className="telem-val" style={{ color: snapshot.drs > 0 ? '#00d27a' : '#5c5c78' }}>
                            {snapshot.drs.toFixed(0)}%
                        </div>
                        <div className="telem-lbl">DRS</div>
                    </div>
                </div>

                {/* Sector times */}
                {snapshot.sectors.length > 0 && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
                        {snapshot.sectors.map((s, i) => (
                            <div key={i} className="telem-cell" style={{ flex: 1 }}>
                                <div className="telem-val" style={{ fontSize: 13 }}>
                                    {s > 0 ? s.toFixed(3) : '—'}
                                </div>
                                <div className="telem-lbl">S{i + 1}</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
