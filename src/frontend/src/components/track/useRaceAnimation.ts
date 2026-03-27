/**
 * useRaceAnimation
 *
 * Bridges Zustand `playbackTime` → a mutable ref that CarsLayer reads
 * on every animation frame without triggering React re-renders.
 *
 * The caller may also override `currentTime` externally (e.g. from a
 * video player scrub event) by writing directly to sessionTimeRef.current.
 */
import { useEffect, useRef } from 'react';
import { useLapPlaybackStore } from '../../stores/lapPlaybackStore';

export function useRaceAnimation() {
    const sessionTimeRef = useRef(0);

    // Keep sessionTimeRef in sync with Zustand playbackTime on every RAF tick.
    // This avoids subscribing CarsLayer to React state which would cause re-renders.
    useEffect(() => {
        let rafId: number;

        const tick = () => {
            sessionTimeRef.current = useLapPlaybackStore.getState().playbackTime;
            rafId = requestAnimationFrame(tick);
        };

        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, []);

    return { sessionTimeRef };
}
