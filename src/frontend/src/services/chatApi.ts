export type ChatRole = 'user' | 'assistant';

export interface ChatApiMessage {
    role: ChatRole;
    content: string;
}

export interface ChatStreamRequest {
    messages: ChatApiMessage[];
    session_id?: string;
    top_k?: number;
    season?: number;
    event_name?: string;
    category?: string;
    allow_llm?: boolean;
    live_context?: Record<string, unknown>;
}

export interface ChatSource {
    id: string | number;
    rank: number;
    title?: string | null;
    source: string;
    category: string;
    season?: number | null;
    event_name?: string | null;
    score: number;
}

interface ChatStreamHandlers {
    onSources?: (sources: ChatSource[]) => void;
    onDelta?: (text: string) => void;
    onDone?: () => void;
    onError?: (message: string) => void;
}

export async function streamChat(
    payload: ChatStreamRequest,
    handlers: ChatStreamHandlers,
): Promise<void> {
    const res = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `${res.status} ${res.statusText}`);
    }
    if (!res.body) {
        throw new Error('No stream body returned by chat API');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';

        for (const block of blocks) {
            const dataLines = block
                .split('\n')
                .filter(line => line.startsWith('data:'))
                .map(line => line.slice(5).trim());
            if (!dataLines.length) continue;

            const raw = dataLines.join('\n');
            try {
                const evt = JSON.parse(raw) as { type: string; [k: string]: unknown };
                if (evt.type === 'sources') {
                    handlers.onSources?.((evt.sources as ChatSource[]) || []);
                } else if (evt.type === 'delta') {
                    handlers.onDelta?.(String(evt.text || ''));
                } else if (evt.type === 'done') {
                    handlers.onDone?.();
                } else if (evt.type === 'error') {
                    handlers.onError?.(String(evt.message || 'Chat stream failed'));
                }
            } catch {
                handlers.onError?.('Malformed chat stream event');
            }
        }
    }
}
