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
import { ChatPanel } from './components/chat/ChatPanel';
import { ChatFab } from './components/chat/ChatFab';
import { StrategyPanel } from './components/chat/StrategyPanel';
import './index.css';

type AssistantMode = 'pitcrew' | 'strategy';

export default function App() {
    const metadata = useSessionStore(s => s.metadata);
    const trackReplay = useSessionStore(s => s.trackReplay);
    const loading = useSessionStore(s => s.loading);
    const error = useSessionStore(s => s.error);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const [showTop, setShowTop] = useState(false);
    const [chatOpen, setChatOpen] = useState(false);
    const [assistantMode, setAssistantMode] = useState<AssistantMode>('pitcrew');

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
            {/* Main Content Area */}
            <main className="bento-main">
                {/* Top Control Hub */}
                <div className="bento-header">
                    <Header />
                </div>

                {metadata && totalLaps > 0 && trackReplay ? (
                    <>
                        {/* Video Stage (7 cols) */}
                        <div className="bento-video">
                            <VideoPlayer />
                        </div>

                        {/* Leaderboard (4 cols) */}
                        <div className={`bento-leaderboard ${chatOpen ? 'chat-active' : ''}`}>
                            {chatOpen
                                ? (assistantMode === 'pitcrew' ? <ChatPanel open={chatOpen} /> : <StrategyPanel />)
                                : <Leaderboard />}
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

            <ChatFab
                open={chatOpen}
                mode={assistantMode}
                onToggleOpen={() => setChatOpen(v => !v)}
                onModeChange={setAssistantMode}
            />

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
