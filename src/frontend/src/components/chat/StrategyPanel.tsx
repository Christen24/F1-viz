import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useLapPlaybackStore } from '../../stores/lapPlaybackStore';
import { useSessionStore } from '../../stores/sessionStore';
import { streamChat, type ChatSource } from '../../services/chatApi';

type StrategyMessage = {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    pending?: boolean;
    sources?: ChatSource[];
};

const QUICK_PROMPTS = [
    'What if a safety car comes out now?',
    'Should the leader pit this lap?',
    'Best tyre strategy for remaining laps?',
];

export function StrategyPanel() {
    const metadata = useSessionStore((s) => s.metadata);
    const currentLap = useLapPlaybackStore((s) => s.currentLap);
    const lapData = useLapPlaybackStore((s) => s.lapData);

    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [messages, setMessages] = useState<StrategyMessage[]>([
        {
            id: 'strategy-welcome',
            role: 'assistant',
            content: 'Prediction Model online. Ask a what-if condition for the ongoing race.',
            pending: false,
            sources: [],
        },
    ]);

    const historyRef = useRef<HTMLDivElement>(null);
    const endRef = useRef<HTMLDivElement>(null);
    const prevCountRef = useRef(messages.length);
    const last = messages[messages.length - 1];
    const lastSig = `${last?.id ?? ''}:${last?.content?.length ?? 0}`;

    const liveContext = useMemo(() => {
        const lap = lapData[Math.max(0, currentLap - 1)];
        const leader = lap?.leader ?? null;
        const top3 = Object.entries(lap?.positions ?? {})
            .map(([code, pos]) => [code, Number(pos)] as const)
            .filter(([, pos]) => Number.isFinite(pos))
            .sort((a, b) => a[1] - b[1])
            .slice(0, 3)
            .map(([code, pos]) => `P${pos} ${code}`);
        return {
            mode: 'strategy',
            session_id: metadata?.session_id,
            race: metadata?.gp,
            current_lap: currentLap,
            total_laps: metadata?.total_laps ?? lapData.length ?? 0,
            leader,
            top3,
        };
    }, [metadata, currentLap, lapData]);

    useEffect(() => {
        const el = historyRef.current;
        if (!el) return;
        const countIncreased = messages.length > prevCountRef.current;
        prevCountRef.current = messages.length;
        el.scrollTo({ top: el.scrollHeight, behavior: countIncreased ? 'smooth' : 'auto' });
        endRef.current?.scrollIntoView({ behavior: countIncreased ? 'smooth' : 'auto', block: 'end' });
    }, [messages.length, lastSig]);

    const ask = async (raw: string) => {
        const query = raw.trim();
        if (!query || sending) return;

        const userId = `su-${Date.now()}`;
        const assistantId = `sa-${Date.now() + 1}`;

        const conversation = [
            ...messages.filter((m) => !m.pending).map((m) => ({ role: m.role, content: m.content })),
            { role: 'user' as const, content: query },
        ];

        setInput('');
        setSending(true);
        setMessages((prev) => [
            ...prev,
            { id: userId, role: 'user', content: query },
            { id: assistantId, role: 'assistant', content: '', pending: true, sources: [] },
        ]);

        try {
            let capturedSources: ChatSource[] = [];
            await streamChat(
                {
                    messages: conversation,
                    session_id: metadata?.session_id,
                    season: metadata?.year,
                    event_name: metadata?.gp,
                    top_k: 8,
                    category: 'strategy',
                    live_context: liveContext,
                },
                {
                    onSources: (sources) => {
                        capturedSources = sources;
                        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, sources } : m)));
                    },
                    onDelta: (delta) => {
                        setMessages((prev) =>
                            prev.map((m) => (m.id === assistantId ? { ...m, content: `${m.content}${delta}` } : m)),
                        );
                    },
                    onDone: () => {
                        setMessages((prev) =>
                            prev.map((m) =>
                                m.id === assistantId ? { ...m, pending: false, sources: capturedSources } : m,
                            ),
                        );
                    },
                    onError: (message) => {
                        setMessages((prev) =>
                            prev.map((m) =>
                                m.id === assistantId
                                    ? { ...m, pending: false, content: m.content || `Prediction error: ${message}` }
                                    : m,
                            ),
                        );
                    },
                },
            );
        } catch (err) {
            const errorText = err instanceof Error ? err.message : 'Failed to reach strategy API';
            setMessages((prev) =>
                prev.map((m) =>
                    m.id === assistantId ? { ...m, pending: false, content: `Prediction error: ${errorText}` } : m,
                ),
            );
        } finally {
            setSending(false);
        }
    };

    const onSubmit = (e: FormEvent) => {
        e.preventDefault();
        void ask(input);
    };

    return (
        <section className="strategy-panel">
            <header className="strategy-panel-header">
                <div>
                    <div className="strategy-panel-kicker">Strategy Lab</div>
                    <h3 className="strategy-panel-title">Prediction Model</h3>
                </div>
                <div className="strategy-live-chip">Lap {currentLap}</div>
            </header>

            <div className="strategy-chat-history" ref={historyRef}>
                {messages.map((m) => (
                    <article key={m.id} className={`strategy-chat-row ${m.role === 'user' ? 'user' : 'assistant'}`}>
                        <div className={`strategy-chat-bubble ${m.role === 'user' ? 'user' : 'assistant'}`}>
                            {m.content || (m.pending ? 'Simulating scenario...' : '')}
                        </div>
                        {m.role === 'assistant' && !m.pending && (
                            <div className="strategy-chat-meta">
                                <span className="strategy-meta-chip">Using live race context</span>
                                <span className="strategy-meta-chip">Leader: {liveContext.leader ?? '--'}</span>
                                {(m.sources || []).slice(0, 2).map((s) => (
                                    <span key={`${m.id}-${s.id}`} className="strategy-meta-chip">
                                        {s.category}
                                    </span>
                                ))}
                            </div>
                        )}
                    </article>
                ))}
                <div ref={endRef} aria-hidden="true" />
            </div>

            <div className="strategy-chip-row">
                {QUICK_PROMPTS.map((prompt) => (
                    <button key={prompt} type="button" className="strategy-chip" onClick={() => void ask(prompt)} disabled={sending}>
                        {prompt}
                    </button>
                ))}
            </div>

            <form className="strategy-input-row" onSubmit={onSubmit}>
                <input
                    className="strategy-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a what-if condition (SC, pit, tyre, weather)..."
                    disabled={sending}
                />
                <button className="strategy-send" type="submit" disabled={sending || !input.trim()}>
                    Predict
                </button>
            </form>
        </section>
    );
}
