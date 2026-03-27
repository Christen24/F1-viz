/**
 * VideoPlayer — Stage with embedded Heatmap Timeline & Event Toasts
 */
import { useState, useRef, useEffect } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import { LapPlaybackControls } from './LapPlaybackControls';

type Quality = '360' | '480' | '720' | '1080' | 'best';

const QUALITIES: { value: Quality; label: string }[] = [
    { value: '360', label: '360p' },
    { value: '480', label: '480p' },
    { value: '720', label: '720p' },
    { value: '1080', label: '1080p' },
    { value: 'best', label: 'Best' },
];


export function VideoPlayer() {
    const metadata = useSessionStore(s => s.metadata);
    const vs = metadata?.video_source;
    const [mode, setMode] = useState<'hl' | 'fr'>('hl');
    const [playing, setPlaying] = useState(false);
    const [quality, setQuality] = useState<Quality>('720');
    const [showQMenu, setShowQMenu] = useState(false);
    const [downloading, setDownloading] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);

    const vid = vs?.video_id ?? '';
    const embedUrl = vs?.embed_url ?? '';
    const gpLabel = metadata ? `${metadata.gp} ${metadata.year}` : '';

    useEffect(() => { setPlaying(false); setDownloading(false); }, [vid]);

    useEffect(() => {
        if (!showQMenu) return;
        const close = () => setShowQMenu(false);
        document.addEventListener('click', close);
        return () => document.removeEventListener('click', close);
    }, [showQMenu]);

    const streamUrl = vid ? `/api/video/stream/${vid}?q=${quality}` : '';

    const startPlay = async () => {
        if (!vid) return;
        setDownloading(true);
        try {
            const r = await fetch(`/api/video/download-status/${vid}?q=${quality}`);
            const d = await r.json();
            if (d.cached) {
                setDownloading(false);
                setPlaying(true);
                return;
            }
        } catch { /* ignore */ }
        setPlaying(true);
    };

    const changeQuality = async (q: Quality) => {
        if (q === quality) { setShowQMenu(false); return; }
        const currentTime = videoRef.current?.currentTime || 0;

        setShowQMenu(false);
        setQuality(q);
        setDownloading(true);
        setPlaying(false);

        await new Promise(r => setTimeout(r, 200));
        setPlaying(true);

        setTimeout(() => {
            if (videoRef.current) {
                videoRef.current.currentTime = currentTime;
                setDownloading(false);
            }
        }, 3000);
    };

    return (
        <>
            <div className="vp-stage">
                {/* Live Pulse Indicator */}
                {(playing || embedUrl) && <div className="live-pulse"><div className="blob"></div>Live</div>}



                <div className="vp-screen">
                    {mode === 'hl' ? (
                        playing && vid ? (
                            <>
                                <video
                                    ref={videoRef}
                                    key={`${vid}-${quality}`}
                                    className="vp-video"
                                    controls
                                    autoPlay
                                    playsInline
                                    preload="auto"
                                    onCanPlay={() => setDownloading(false)}
                                    onPlaying={() => setDownloading(false)}
                                >
                                    <source src={streamUrl} type="video/mp4" />
                                </video>

                                {downloading && (
                                    <div className="vp-download-overlay">
                                        <div className="vp-download-spinner" />
                                        <div className="vp-download-text">Preparing {quality === 'best' ? 'best' : quality + 'p'} stream...</div>
                                    </div>
                                )}

                                <div className="vp-quality-badge" onClick={e => { e.stopPropagation(); setShowQMenu(v => !v); }}>
                                    <span className="vp-quality-icon">⚙</span><span>{quality === 'best' ? 'Best' : quality + 'p'}</span>
                                </div>
                                {showQMenu && (
                                    <div className="vp-quality-menu" onClick={e => e.stopPropagation()}>
                                        <div className="vp-quality-title">Video Quality</div>
                                        {QUALITIES.map(q => (
                                            <button
                                                key={q.value}
                                                className={'vp-quality-option' + (q.value === quality ? ' vp-quality-active' : '')}
                                                onClick={() => changeQuality(q.value)}
                                            >
                                                <span>{q.label}</span>
                                                {q.value === quality && <span className="vp-quality-check">✓</span>}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="vp-poster" onClick={downloading ? undefined : () => startPlay()}>
                                {vid ? (
                                    <>
                                        <img
                                            className="vp-thumb"
                                            src={'https://img.youtube.com/vi/' + vid + '/hqdefault.jpg'}
                                            alt={gpLabel + ' thumbnail'}
                                            onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                        />
                                        {downloading ? (
                                            <>
                                                <div className="vp-download-spinner" />
                                                <div className="vp-label">Preparing stream...</div>
                                            </>
                                        ) : (
                                            <>
                                                <div className="vp-play-circle"><div className="vp-play-triangle"></div></div>
                                                <div className="vp-label">{gpLabel} Highlights</div>
                                            </>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        <div className="vp-label">{metadata ? gpLabel : 'Race Highlights'}</div>
                                        <div className="vp-sub">{metadata ? 'No highlight video found' : 'Load a session to begin'}</div>
                                    </>
                                )}
                            </div>
                        )
                    ) : (
                        embedUrl ? (
                            <iframe
                                key={embedUrl}
                                className="vp-iframe"
                                src={embedUrl}
                                title={gpLabel + ' Full Race'}
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
                                allowFullScreen
                                onLoad={() => setPlaying(true)}
                            />
                        ) : (
                            <div className="vp-poster">
                                <div className="vp-label">{metadata ? gpLabel : 'Full Race'}</div>
                                <div className="vp-sub">{metadata ? 'No replay available' : 'Select a race'}</div>
                            </div>
                        )
                    )}
                </div>

                {/* Integrated Scrub-Sync Timeline */}
                <LapPlaybackControls />
            </div>
        </>
    );
}
