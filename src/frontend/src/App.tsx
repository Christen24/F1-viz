/**
 * App.tsx — F1 Race Intelligence Platform (2026 Bento Redesign)
 */
import { useEffect, useState, useCallback } from 'react';
import { useSessionStore } from './stores/sessionStore';
import { useLapPlaybackStore } from './stores/lapPlaybackStore';
import { Header } from './components/Header';
import { VideoPlayer } from './components/VideoPlayer';
import { LapPlaybackControls } from './components/LapPlaybackControls';
import { GapChart } from './components/GapChart';
import { PositionChart } from './components/PositionChart';
import { TireStrategy } from './components/TireStrategy';
import { Leaderboard } from './components/Leaderboard';
import { TrackMap } from './components/track/TrackMap';
import './index.css';

export default function App() {
    const metadata = useSessionStore(s => s.metadata);
    const loading = useSessionStore(s => s.loading);
    const error = useSessionStore(s => s.error);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const [showTop, setShowTop] = useState(false);

    // Show back-to-top button after scrolling down
    useEffect(() => {
        const handler = () => setShowTop(window.scrollY > 400);
        window.addEventListener('scroll', handler, { passive: true });
        return () => window.removeEventListener('scroll', handler);
    }, []);

    const scrollToTop = useCallback(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, []);

    // Keyboard shortcuts for lap playback
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            const tgt = e.target as HTMLElement;
            if (tgt.tagName === 'INPUT' || tgt.tagName === 'SELECT' || tgt.tagName === 'TEXTAREA') return;

            const lapStore = useLapPlaybackStore.getState();
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    lapStore.togglePlay();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    lapStore.prevLap();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    lapStore.nextLap();
                    break;
                default:
                    if (e.key >= '1' && e.key <= '9') {
                        const frac = parseInt(e.key) / 10;
                        const lap = Math.max(1, Math.round(frac * lapStore.totalLaps));
                        lapStore.setLap(lap);
                    }
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    return (
        <div className="bento-layout">
            {/* 1 Col Global Sidebar */}
            <aside className="global-sidebar">
                <button className="nav-btn active" title="Home">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>
                </button>
                <button className="nav-btn" title="Telemetry">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                </button>
                <button className="nav-btn" title="Compare">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></svg>
                </button>
                <button className="nav-btn" title="Settings">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
                </button>
            </aside>

            {/* 11 Col Main Content Area */}
            <main className="bento-main">
                {/* Top Control Hub */}
                <div className="bento-header">
                    <Header />
                </div>

                {metadata && totalLaps > 0 ? (
                    <>
                        {/* Video Stage (7 cols) */}
                        <div className="bento-video">
                            <VideoPlayer />
                        </div>

                        {/* Leaderboard (4 cols) */}
                        <div className="bento-leaderboard">
                            <Leaderboard />
                        </div>

                        {/* Bento Data Row (4+4+3 cols = 11 cols) */}
                        <div className="bento-track">
                            <TrackMap />
                        </div>

                        <div className="bento-gap">
                            <GapChart />
                        </div>

                        <div className="bento-tire">
                            <TireStrategy />
                        </div>
                    </>
                ) : metadata ? (
                    <div className="empty" style={{ gridColumn: 'span 11' }}>
                        <div className="ico">📊</div>
                        <div>Loading lap data…</div>
                    </div>
                ) : !loading ? (
                    <div className="empty" style={{ gridColumn: 'span 11' }}>
                        <div className="ico">🏎️</div>
                        <div>Select a race from the header to begin</div>
                    </div>
                ) : null}
            </main>

            {/* Back to top & Toasts */}
            {showTop && (
                <button className="back-to-top" onClick={scrollToTop} aria-label="Back to top" title="Back to top">
                    ↑
                </button>
            )}

            {loading && (
                <div className="toast">
                    <span className="spin" />
                    <div>
                        <div>Loading race data…</div>
                        <div style={{ fontSize: 10, color: 'var(--color-text-muted)' }}>First load ≈ 1–2 min</div>
                    </div>
                </div>
            )}

            {error && (
                <div className="toast toast-error">
                    ⚠ {error}
                    <button
                        onClick={() => useSessionStore.getState().setError(null)}
                        style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontWeight: 700, marginLeft: 8 }}
                    >✕</button>
                </div>
            )}
        </div>
    );
}
