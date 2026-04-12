type AssistantMode = 'pitcrew' | 'strategy';

interface ChatFabProps {
    open: boolean;
    mode: AssistantMode;
    onToggleOpen: () => void;
    onModeChange: (mode: AssistantMode) => void;
}

export function ChatFab({ open, mode, onToggleOpen, onModeChange }: ChatFabProps) {
    return (
        <div className="chat-fab-wrap">
            {open && (
                <div className="chat-fab-menu" role="tablist" aria-label="Assistant mode switch">
                    <button
                        type="button"
                        className={`chat-fab-mode ${mode === 'pitcrew' ? 'active' : ''}`}
                        onClick={() => onModeChange('pitcrew')}
                    >
                        AI Pit Crew
                    </button>
                    <button
                        type="button"
                        className={`chat-fab-mode ${mode === 'strategy' ? 'active' : ''}`}
                        onClick={() => onModeChange('strategy')}
                    >
                        Prediction
                    </button>
                </div>
            )}

            <button
                className={`chat-fab ${open ? 'open' : ''}`}
                onClick={onToggleOpen}
                aria-label={open ? 'Close assistant panel' : 'Open assistant panel'}
                title={open ? 'Close Assistant Panel' : 'Open Assistant Panel'}
            >
                <span className="chat-fab-icon" aria-hidden="true">
                    {open ? (
                        <span className="chat-fab-close">×</span>
                    ) : (
                        <img src="/steering-wheel.svg" alt="AI" className="chat-fab-img" />
                    )}
                </span>
            </button>
        </div>
    );
}
