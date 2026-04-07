import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useSessionStore } from '../../stores/sessionStore';
import { useLapPlaybackStore } from '../../stores/lapPlaybackStore';
import { streamChat, type ChatSource } from '../../services/chatApi';

type PanelMessage = {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    pending?: boolean;
    sources?: ChatSource[];
};

const QUICK_PROMPTS = [
    'Who is leading right now?',
    'Give me a lap summary for this race.',
    'How is the current leader performing?',
];

interface ChatPanelProps {
    open: boolean;
}

export function ChatPanel({ open }: ChatPanelProps) {
    const metadata = useSessionStore((s) => s.metadata);
    const currentLap = useLapPlaybackStore((s) => s.currentLap);
    const lapData = useLapPlaybackStore((s) => s.lapData);

    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [messages, setMessages] = useState<PanelMessage[]>([
        {
            id: 'welcome',
            role: 'assistant',
            content: 'AI Pit Crew online. Ask me about race stats, laps, drivers, or strategy context.',
        },
    ]);

    const listRef = useRef<HTMLDivElement>(null);
    const endRef = useRef<HTMLDivElement>(null);
    const prevMsgCountRef = useRef(messages.length);
    const leader = lapData[currentLap - 1]?.leader ?? null;
    const lastMessage = messages[messages.length - 1];
    const lastMessageSig = `${lastMessage?.id ?? ''}:${lastMessage?.content?.length ?? 0}`;

    useEffect(() => {
        const el = listRef.current;
        if (!el) return;

        const countIncreased = messages.length > prevMsgCountRef.current;
        prevMsgCountRef.current = messages.length;
        if (!countIncreased && !open) return;

        el.scrollTo({
            top: el.scrollHeight,
            behavior: countIncreased ? 'smooth' : 'auto',
        });
        endRef.current?.scrollIntoView({
            behavior: countIncreased ? 'smooth' : 'auto',
            block: 'end',
        });
    }, [messages.length, lastMessageSig, open]);

    const contextTags = useMemo(() => {
        const tags = ['Using live race data'];
        if (currentLap > 0) tags.push(`Lap ${currentLap}`);
        if (leader) tags.push(`Leader: ${leader}`);
        return tags;
    }, [currentLap, leader]);

    const send = async (text: string) => {
        const trimmed = text.trim();
        if (!trimmed || sending) return;

        const userMsg: PanelMessage = {
            id: `u-${Date.now()}`,
            role: 'user',
            content: trimmed,
        };
        const assistantId = `a-${Date.now() + 1}`;

        const conversation = [
            ...messages.filter((m) => !m.pending).map((m) => ({ role: m.role, content: m.content })),
            { role: 'user' as const, content: trimmed },
        ];

        setInput('');
        setSending(true);
        setMessages((prev) => [
            ...prev,
            userMsg,
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
                    top_k: 6,
                    allow_llm: false,
                    live_context: {
                        session_id: metadata?.session_id,
                        race: metadata?.gp,
                        current_lap: currentLap,
                        leader,
                    },
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
                                m.id === assistantId
                                    ? { ...m, pending: false, sources: capturedSources }
                                    : m,
                            ),
                        );
                    },
                    onError: (message) => {
                        setMessages((prev) =>
                            prev.map((m) =>
                                m.id === assistantId
                                    ? {
                                        ...m,
                                        pending: false,
                                        content: m.content || `I hit an error: ${message}`,
                                    }
                                    : m,
                            ),
                        );
                    },
                },
            );
        } catch (err) {
            const errorText = err instanceof Error ? err.message : 'Failed to reach chat API';
            setMessages((prev) =>
                prev.map((m) =>
                    m.id === assistantId ? { ...m, pending: false, content: `I hit an error: ${errorText}` } : m,
                ),
            );
        } finally {
            setSending(false);
        }
    };

    const onSubmit = (e: FormEvent) => {
        e.preventDefault();
        void send(input);
    };

    return (
        <section className={`chat-panel ${open ? 'open' : ''}`} aria-hidden={!open}>
            <header className="chat-panel-header">
                <div>
                    <div className="chat-panel-kicker">Race Assistant</div>
                    <h3 className="chat-panel-title">AI Pit Crew</h3>
                </div>
            </header>

            <div className="chat-panel-body">
                <div className="chat-history" ref={listRef}>
                    {messages.map((msg) => (
                        <article key={msg.id} className={`chat-bubble-row ${msg.role === 'user' ? 'user' : 'assistant'}`}>
                            <div className={`chat-bubble ${msg.role === 'user' ? 'user' : 'assistant'}`}>
                                {msg.content || (msg.pending ? 'Thinking...' : '')}
                                {msg.pending && <span className="chat-caret" aria-hidden="true">▌</span>}
                            </div>
                            {msg.role === 'assistant' && !msg.pending && (
                                <div className="chat-context-chips">
                                    {contextTags.map((tag) => (
                                        <span key={`${msg.id}-${tag}`} className="chat-context-chip">{tag}</span>
                                    ))}
                                    {(msg.sources || []).slice(0, 2).map((src) => (
                                        <span key={`${msg.id}-src-${src.id}`} className="chat-context-chip source">
                                            {src.category}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </article>
                    ))}
                    <div ref={endRef} aria-hidden="true" />
                </div>

                <div className="chat-suggestions">
                    {QUICK_PROMPTS.map((prompt) => (
                        <button
                            key={prompt}
                            type="button"
                            className="chat-chip"
                            onClick={() => void send(prompt)}
                            disabled={sending}
                        >
                            {prompt}
                        </button>
                    ))}
                </div>
            </div>

            <form className="chat-input-row" onSubmit={onSubmit}>
                <input
                    className="chat-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={sending}
                    placeholder="Ask about race stats, drivers, laps..."
                />
                <button className="chat-send" type="submit" disabled={sending || !input.trim()}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M22 2 11 13" />
                        <path d="M22 2 15 22 11 13 2 9 22 2z" />
                    </svg>
                </button>
            </form>
        </section>
    );
}
