interface ChatFabProps {
    open: boolean;
    onClick: () => void;
}

export function ChatFab({ open, onClick }: ChatFabProps) {
    return (
        <button
            className={`chat-fab ${open ? 'open' : ''}`}
            onClick={onClick}
            aria-label={open ? 'Close AI chat assistant' : 'Open AI chat assistant'}
            title={open ? 'Close Race Assistant' : 'Open Race Assistant'}
        >
            <span className="chat-fab-icon" aria-hidden="true">
                {open ? '×' : 'AI'}
            </span>
        </button>
    );
}
