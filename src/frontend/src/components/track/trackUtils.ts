import type { TrackPoint } from '../../types/session';

const VIEWBOX_WIDTH = 1000;
const VIEWBOX_HEIGHT = 620;
const TRACK_PADDING = 72;

export interface NormalizedTrack {
    points: TrackPoint[];
    width: number;
    height: number;
}

export function lerp(start: number, end: number, t: number): number {
    return start + (end - start) * t;
}

export function buildTrackPath(points: TrackPoint[]): string {
    if (points.length === 0) return '';
    const commands = [`M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`];
    for (let i = 1; i < points.length; i += 1) {
        commands.push(`L ${points[i].x.toFixed(2)} ${points[i].y.toFixed(2)}`);
    }
    commands.push('Z');
    return commands.join(' ');
}

export function normalizeTrack(points: TrackPoint[]): NormalizedTrack {
    if (points.length === 0) {
        return { points: [], width: VIEWBOX_WIDTH, height: VIEWBOX_HEIGHT };
    }

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (const point of points) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);
        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
    }

    const rawWidth = Math.max(maxX - minX, 1);
    const rawHeight = Math.max(maxY - minY, 1);
    const scale = Math.min(
        (VIEWBOX_WIDTH - TRACK_PADDING * 2) / rawWidth,
        (VIEWBOX_HEIGHT - TRACK_PADDING * 2) / rawHeight,
    );

    const offsetX = (VIEWBOX_WIDTH - rawWidth * scale) / 2;
    const offsetY = (VIEWBOX_HEIGHT - rawHeight * scale) / 2;

    return {
        width: VIEWBOX_WIDTH,
        height: VIEWBOX_HEIGHT,
        points: points.map((point) => ({
            x: (point.x - minX) * scale + offsetX,
            y: VIEWBOX_HEIGHT - ((point.y - minY) * scale + offsetY),
        })),
    };
}

export function getCarPosition(trackPoints: TrackPoint[], distance: number, trackLength: number): TrackPoint {
    if (trackPoints.length === 0) return { x: 0, y: 0 };
    if (trackPoints.length === 1 || trackLength <= 0) return trackPoints[0];

    const wrappedDistance = ((distance % trackLength) + trackLength) % trackLength;
    const progress = wrappedDistance / trackLength;
    const scaledIndex = progress * trackPoints.length;
    const index = Math.floor(scaledIndex) % trackPoints.length;
    const nextIndex = (index + 1) % trackPoints.length;
    const t = scaledIndex - Math.floor(scaledIndex);
    const current = trackPoints[index];
    const next = trackPoints[nextIndex];

    return {
        x: lerp(current.x, next.x, t),
        y: lerp(current.y, next.y, t),
    };
}

export function interpolateDistance(start: number, end: number, t: number, trackLength: number): number {
    if (trackLength <= 0) return lerp(start, end, t);

    let delta = end - start;
    if (Math.abs(delta) > trackLength / 2) {
        delta = delta > 0 ? delta - trackLength : delta + trackLength;
    }

    const value = start + delta * t;
    return ((value % trackLength) + trackLength) % trackLength;
}

export function upperBound(values: number[], target: number): number {
    let low = 0;
    let high = values.length;

    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (values[mid] <= target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}
