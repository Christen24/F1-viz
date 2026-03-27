/**
 * InsightsPanel — AI-generated NL insights with categories
 * 
 * Expanded view with better readability and category styling.
 */
import { useSessionStore } from '../stores/sessionStore';

const CAT_COLORS: Record<string, { bg: string; color: string; icon: string }> = {
    race: { bg: 'rgba(225, 6, 0, 0.15)', color: '#e10600', icon: '🏁' },
    strategy: { bg: 'rgba(255, 165, 0, 0.12)', color: '#ffa500', icon: '🧠' },
    pace: { bg: 'rgba(0, 180, 255, 0.12)', color: '#00b4ff', icon: '⚡' },
    tire: { bg: 'rgba(46, 204, 113, 0.12)', color: '#2ecc71', icon: '🔧' },
    overtake: { bg: 'rgba(155, 89, 182, 0.12)', color: '#9b59b6', icon: '🔄' },
    incident: { bg: 'rgba(231, 76, 60, 0.12)', color: '#e74c3c', icon: '⚠️' },
    default: { bg: 'rgba(255, 255, 255, 0.06)', color: '#8888aa', icon: '📊' },
};

export function InsightsPanel() {
    const insights = useSessionStore(s => s.insights);

    if (!insights.length) return (
        <div className="card">
            <div className="panel-hdr">
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="panel-icon">💡</span> Insights
                </span>
            </div>
            <div className="panel-body">
                <div className="empty" style={{ padding: 20, textAlign: 'center' }}>
                    <div style={{ fontSize: 24, marginBottom: 6 }}>🤖</div>
                    <div style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>Awaiting analysis</div>
                </div>
            </div>
        </div>
    );

    return (
        <div className="card insights-panel">
            <div className="panel-hdr">
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="panel-icon">💡</span> Insights
                </span>
                <span className="insights-count">{insights.length}</span>
            </div>
            <div className="panel-body flush insights-body">
                {insights.map((ins, i) => {
                    const cat = CAT_COLORS[ins.category] || CAT_COLORS.default;
                    return (
                        <div key={i} className="ins-card" role="article">
                            <div className="ins-header">
                                <span className="ins-cat-badge" style={{ background: cat.bg, color: cat.color }}>
                                    {cat.icon} {ins.category}
                                </span>
                            </div>
                            <div className="ins-text">{ins.text}</div>
                            {ins.related_drivers && ins.related_drivers.length > 0 && (
                                <div className="ins-drivers">
                                    {ins.related_drivers.map((d, j) => (
                                        <span key={j} className="ins-driver-tag">{d}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
