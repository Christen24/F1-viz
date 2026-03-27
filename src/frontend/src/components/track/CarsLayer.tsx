import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import type { TrackPoint, TrackReplayDriver, TrackReplayFrame } from '../../types/session';
import { getCarPosition, interpolateDistance, upperBound } from './trackUtils';

const VIEWBOX_W = 1000;
const VIEWBOX_H = 620;
const PADDING = 72;

interface CarsLayerProps {
    drivers: TrackReplayDriver[];
    frames: TrackReplayFrame[];
    sessionTimeRef: MutableRefObject<number>;
    rawTrackPoints: TrackPoint[];
    normalizedTrackPoints: TrackPoint[];
    trackLength: number;
    leaderId?: string;
}

interface BoundingBox {
    minX: number;
    minY: number;
    scale: number;
    offsetX: number;
    offsetY: number;
}

function computeBoundingBox(rawPoints: TrackPoint[]): BoundingBox | null {
    if (!rawPoints.length) return null;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const p of rawPoints) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    }

    const rawW = Math.max(maxX - minX, 1);
    const rawH = Math.max(maxY - minY, 1);
    const scale = Math.min(
        (VIEWBOX_W - PADDING * 2) / rawW,
        (VIEWBOX_H - PADDING * 2) / rawH,
    );
    const offsetX = (VIEWBOX_W - rawW * scale) / 2;
    const offsetY = (VIEWBOX_H - rawH * scale) / 2;

    return { minX, minY, scale, offsetX, offsetY };
}

function mapXY(x: number, y: number, box: BoundingBox): TrackPoint {
    return {
        x: (x - box.minX) * box.scale + box.offsetX,
        y: VIEWBOX_H - ((y - box.minY) * box.scale + box.offsetY),
    };
}

function getDistanceValue(value: TrackReplayFrame['positions'][string]): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (value && typeof value === 'object' && typeof value.d === 'number' && Number.isFinite(value.d)) {
        return value.d;
    }
    return null;
}

function getXYValue(value: TrackReplayFrame['positions'][string]): TrackPoint | null {
    if (!value || typeof value !== 'object') return null;
    if (typeof value.x !== 'number' || typeof value.y !== 'number') return null;
    if (!Number.isFinite(value.x) || !Number.isFinite(value.y)) return null;
    return { x: value.x, y: value.y };
}

export function CarsLayer({
    drivers,
    frames,
    sessionTimeRef,
    rawTrackPoints,
    normalizedTrackPoints,
    trackLength,
    leaderId,
}: CarsLayerProps) {
    const markerRefs = useRef<Record<string, SVGGElement | null>>({});
    const frameTimes = useMemo(() => frames.map((f) => f.t), [frames]);
    const boundingBox = useMemo(() => computeBoundingBox(rawTrackPoints), [rawTrackPoints]);

    useEffect(() => {
        if (!drivers.length || !frames.length) {
            return undefined;
        }

        let rafId = 0;

        const updateCars = () => {
            const now = sessionTimeRef.current;
            const hi = upperBound(frameTimes, now);
            const i0 = Math.max(0, hi - 1);
            const i1 = Math.min(frames.length - 1, hi);
            const f0 = frames[i0];
            const f1 = frames[i1];

            const dt = Math.max(f1.t - f0.t, 0.001);
            const alpha = Math.min(1, Math.max(0, (now - f0.t) / dt));

            for (const driver of drivers) {
                const marker = markerRefs.current[driver.id];
                if (!marker) continue;

                const d0 = getDistanceValue(f0.positions[driver.id]);
                const d1 = getDistanceValue(f1.positions[driver.id]);
                let pos: TrackPoint | null = null;

                // Preferred path: lightweight distance interpolation on normalized track.
                if (d0 !== null && d1 !== null && normalizedTrackPoints.length >= 2 && trackLength > 0) {
                    const distance = interpolateDistance(d0, d1, alpha, trackLength);
                    pos = getCarPosition(normalizedTrackPoints, distance, trackLength);
                } else if (boundingBox) {
                    // Compatibility path: direct x/y interpolation if replay payload includes it.
                    const p0 = getXYValue(f0.positions[driver.id]);
                    const p1 = getXYValue(f1.positions[driver.id]);
                    if (p0 && p1) {
                        const rawX = p0.x + (p1.x - p0.x) * alpha;
                        const rawY = p0.y + (p1.y - p0.y) * alpha;
                        pos = mapXY(rawX, rawY, boundingBox);
                    }
                }

                if (!pos) {
                    marker.style.opacity = '0';
                    continue;
                }

                marker.setAttribute('transform', `translate(${pos.x.toFixed(2)} ${pos.y.toFixed(2)})`);
                marker.style.opacity = '1';
            }

            rafId = requestAnimationFrame(updateCars);
        };

        rafId = requestAnimationFrame(updateCars);
        return () => cancelAnimationFrame(rafId);
    }, [
        boundingBox,
        drivers,
        frameTimes,
        frames,
        normalizedTrackPoints,
        sessionTimeRef,
        trackLength,
    ]);

    return (
        <g className="track-cars-layer">
            {drivers.map((driver) => (
                <g
                    key={driver.id}
                    ref={(node) => { markerRefs.current[driver.id] = node; }}
                    className={`track-car-marker${leaderId === driver.id ? ' leader' : ''}`}
                    style={{ opacity: 0 }}
                >
                    {leaderId === driver.id && (
                        <>
                            <circle
                                className="track-car-leader-ring"
                                r="10.5"
                                fill="none"
                                stroke="#ffd54a"
                                strokeWidth="1.8"
                            />
                            <g className="track-p1-badge" transform="translate(11 -16)">
                                <rect rx="3" ry="3" width="19" height="12" />
                                <text x="9.5" y="8">P1</text>
                            </g>
                        </>
                    )}
                    <circle
                        className="track-car-dot"
                        r="7"
                        fill={driver.color.startsWith('#') ? driver.color : `#${driver.color}`}
                        stroke="rgba(255,255,255,0.92)"
                        strokeWidth="1.6"
                    />
                    <text
                        className="track-car-label"
                        x="10"
                        y="4"
                        fontSize="10"
                        fill="#fff"
                        fontWeight="700"
                        style={{ pointerEvents: 'none', userSelect: 'none' }}
                    >
                        {driver.id}
                    </text>
                </g>
            ))}
        </g>
    );
}
