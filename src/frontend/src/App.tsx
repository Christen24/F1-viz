/**
 * App.tsx — F1 Race Intelligence Platform (2026 Bento Redesign)
 * Panel resize: drag the left edge of the chat/leaderboard column
 */
import { useEffect, useState, useCallback, useRef } from 'react';
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

const PANEL_WIDTH_KEY = 'f1viz_panel_width';
const PANEL_MIN_PX = 280;
const PANEL_MAX_PX = 900;
const PANEL_DEFAULT_PX = 420;

function loadPanelWidth(): number {
    try {
        const v = localStorage.getItem(PANEL_WIDTH_KEY);
        if (v) {
            const n = parseInt(v, 10);
            if (n >= PANEL_MIN_PX && n <= PANEL_MAX_PX) return n;
        }
    } catch { /* ignore */ }
    return PANEL_DEFAULT_PX;
}

export default function App() {
    const metadata = useSessionStore(s => s.metadata);
    const trackReplay = useSessionStore(s => s.trackReplay);
    const loading = useSessionStore(s => s.loading);
    const error = useSessionStore(s => s.error);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const [showTop, setShowTop] = useState(false);
    const [chatOpen, setChatOpen] = useState(false);
    const [assistantMode, setAssistantMode] = useState<AssistantMode>('pitcrew');

    // ── Drag-to-resize panel width ───────────────────────────────────────────
    const [panelWidth, setPanelWidth] = useState<number>(loadPanelWidth);
    const dragState = useRef<{ startX: number; startW: number } | null>(null);

    const onResizeMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        dragState.current = { startX: e.clientX, startW: panelWidth };
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    }, [panelWidth]);

    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!dragState.current) return;
            // Dragging left edge: moving left increases panel width
            const delta = dragState.current.startX - e.clientX;
            const next = Math.min(PANEL_MAX_PX, Math.max(PANEL_MIN_PX, dragState.current.startW + delta));
            setPanelWidth(next);
        };
        const onUp = () => {
            if (!dragState.current) return;
            dragState.current = null;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            // Persist width
            try { localStorage.setItem(PANEL_WIDTH_KEY, String(panelWidth)); } catch { /* ignore */ }
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        return () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
    }, [panelWidth]);

    // Persist width whenever it settles
    useEffect(() => {
        try { localStorage.setItem(PANEL_WIDTH_KEY, String(panelWidth)); } catch { /* ignore */ }
    }, [panelWidth]);

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
                        {/* Video Stage and Leaderboard Wrapper */}
                        <div style={{ gridColumn: 'span 11', display: 'flex', gap: '16px' }}>
                            {/* Video Stage */}
                            <div className="bento-video" style={chatOpen ? { flex: '1 1 0', minWidth: 0, gridColumn: 'unset' } : { flex: '7 1 0', gridColumn: 'unset' }}>
                                <VideoPlayer />
                            </div>

                            {/* Leaderboard / Chat panel — resizable */}
                            <div
                                className={`bento-leaderboard ${chatOpen ? 'chat-active' : ''}`}
                                style={chatOpen ? { flex: `0 0 ${panelWidth}px`, width: `${panelWidth}px`, minWidth: `${PANEL_MIN_PX}px`, maxWidth: `${PANEL_MAX_PX}px`, gridColumn: 'unset' } : { flex: '4 1 0', gridColumn: 'unset' }}
                            >
                                {/* Drag handle — only visible when chat is open */}
                                {chatOpen && (
                                    <div
                                        className="panel-resize-handle"
                                        onMouseDown={onResizeMouseDown}
                                        title="Drag to resize panel"
                                        aria-hidden="true"
                                    />
                                )}

                                {chatOpen
                                    ? (assistantMode === 'pitcrew'
                                        ? <ChatPanel open={chatOpen} onClose={() => setChatOpen(false)} />
                                        : <StrategyPanel onClose={() => setChatOpen(false)} />)
                                    : <Leaderboard />}
                            </div>
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
