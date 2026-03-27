/**
 * GapChart — Gap to leader per lap, styled after the reference broadcast chart.
 *
 * Design:
 *  - Clean multiline chart, no area fill
 *  - Inline driver labels at the right end of each line
 *  - Subtle dark grid lines
 *  - Y-axis inverted (leader 0 at top, gaps go down) with centered reference
 *  - Full-race X-axis always visible; vertical NOW marker tracks current lap
 */
import ReactECharts from 'echarts-for-react';
import { useLapPlaybackStore } from '../stores/lapPlaybackStore';
import { useSessionStore } from '../stores/sessionStore';
import { useMemo, useRef } from 'react';
import type { LapSummary } from '../types/session';

export function GapChart() {
    const lapData = useLapPlaybackStore(s => s.lapData);
    const currentLap = useLapPlaybackStore(s => s.currentLap);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const selectedDrivers = useSessionStore(s => s.selectedDrivers);
    const soloDriver = useSessionStore(s => s.soloDriver);
    const metadata = useSessionStore(s => s.metadata);
    const chartRef = useRef<ReactECharts>(null);

    const driverColors = useMemo(() => {
        if (!metadata) return {} as Record<string, string>;
        const c: Record<string, string> = {};
        metadata.drivers.forEach(d => {
            c[d.code] = d.team_color.startsWith('#') ? d.team_color : `#${d.team_color}`;
        });
        return c;
    }, [metadata]);

    const activeDrivers = soloDriver ? [soloDriver] : selectedDrivers;

    // Stable Y extents across full race (so axis doesn't jump while scrubbing)
    const maxGap = useMemo(() => {
        if (!lapData.length || !activeDrivers.length) return 60;
        let m = 0;
        for (const lap of lapData) {
            for (const code of activeDrivers) {
                const g = lap.gaps[code];
                if (g !== undefined && g > m) m = g;
            }
        }
        return Math.ceil(Math.max(m * 1.1, 5));
    }, [lapData, activeDrivers]);

    const series = useMemo(() => {
        if (!lapData.length || !activeDrivers.length) return [];

        return activeDrivers.map(code => {
            const color = driverColors[code] || '#aaa';
            const points = lapData
                .slice(0, currentLap)
                .filter((l: LapSummary) => l.gaps[code] !== undefined)
                .map((l: LapSummary) => [l.lap, l.gaps[code]]);

            return {
                name: code,
                type: 'line' as const,
                data: points,
                smooth: 0.3,
                symbol: 'none',
                lineStyle: { width: 2, color },
                itemStyle: { color },
                // Inline label at the last data point (right end of line)
                endLabel: {
                    show: true,
                    formatter: code,
                    color,
                    fontSize: 11,
                    fontFamily: 'Inter',
                    fontWeight: 700,
                    offset: [4, 0],
                },
                emphasis: { disabled: true },
                animationDuration: 400,
                markLine: undefined as any,
            };
        });
    }, [lapData, activeDrivers, driverColors, currentLap]);

    if (!metadata) return null;

    if (!activeDrivers.length) {
        return (
            <div className="gap-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--glass-border)', fontSize: 11, fontWeight: 800, letterSpacing: 1, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
                    Gap to Leader
                </div>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-muted)', fontSize: 13 }}>
                    Select drivers in the classification panel
                </div>
            </div>
        );
    }

    // 1) P1 dashed reference line
    const p1MarkLine = {
        silent: true,
        symbol: 'none',
        lineStyle: { color: 'rgba(225,6,0,0.35)', type: 'dashed', width: 1 },
        label: { show: false },
        data: [{ yAxis: 0 }],
    };

    if (series.length > 0) {
        series[0].markLine = {
            silent: true,
            symbol: 'none',
            data: [{
                yAxis: 0,
                lineStyle: p1MarkLine.lineStyle,
                label: p1MarkLine.label
            }]
        };
    }

    const option = {
        backgroundColor: 'transparent',
        animation: true,
        tooltip: {
            trigger: 'axis' as const,
            backgroundColor: 'rgba(10, 10, 15, 0.96)',
            borderColor: 'rgba(255,255,255,0.08)',
            borderWidth: 1,
            padding: [10, 14],
            borderRadius: 8,
            textStyle: { color: '#fff', fontSize: 11, fontFamily: 'Inter' },
            axisPointer: {
                type: 'line',
                lineStyle: { color: 'rgba(255,255,255,0.2)', width: 1, type: 'dashed' },
            },
            formatter(params: any[]) {
                if (!params?.length) return '';
                const lap = params[0].axisValue;
                let html = `<div style="margin-bottom:6px;font-size:10px;color:#666;font-family:'JetBrains Mono',monospace">LAP ${lap}</div>`;
                const sorted = [...params].sort((a, b) => (a.data[1] ?? 999) - (b.data[1] ?? 999));
                for (const p of sorted) {
                    const gap = p.data[1];
                    if (gap === undefined) continue;
                    const label = gap === 0 ? '<span style="color:#e10600;font-weight:800">LEADER</span>'
                        : `+${gap.toFixed(3)}s`;
                    html += `<div style="display:flex;align-items:center;gap:8px;margin:3px 0">
                            <span style="width:10px;height:3px;border-radius:2px;background:${p.color};display:inline-block"></span>
                            <span style="font-weight:700;min-width:36px;font-family:'JetBrains Mono',monospace;font-size:11px">${p.seriesName}</span>
                            <span style="color:#ccc;font-family:'JetBrains Mono',monospace;font-size:11px">${label}</span>
                        </div>`;
                }
                return html;
            },
        },
        legend: { show: false },
        grid: { left: 52, right: 64, top: 18, bottom: 30 },
        xAxis: {
            type: 'value' as const,
            name: 'Lap',
            nameLocation: 'middle',
            nameGap: 20,
            nameTextStyle: { color: '#4a4a66', fontSize: 10, fontFamily: 'Inter' },
            min: 1,
            max: totalLaps || 'dataMax',
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
            axisTick: { show: false },
            axisLabel: {
                color: '#4a4a66',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
                interval: 4,
            },
            splitLine: {
                show: true,
                lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'solid' },
            },
        },
        yAxis: {
            type: 'value' as const,
            name: 'Gap (s)',
            nameLocation: 'middle',
            nameGap: 40,
            nameRotate: 90,
            nameTextStyle: { color: '#4a4a66', fontSize: 10, fontFamily: 'Inter' },
            inverse: true,
            min: 0,
            max: maxGap,
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: {
                color: '#4a4a66',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
                formatter: (v: number) => v === 0 ? 'LDR' : `+${v}`,
            },
            splitLine: {
                show: true,
                lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'solid' },
            },
        },
        series: series,
    };

    return (
        <div className="gap-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{
                padding: '10px 16px',
                borderBottom: '1px solid var(--glass-border)',
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
                flexShrink: 0,
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: 1.5, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
                        Gap to Leader
                    </span>
                    <span style={{ fontSize: 9, color: '#4a4a66' }}>per lap · seconds</span>
                </div>

                {/* Driver Toggles */}
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {metadata.drivers.map(d => {
                        const isActive = activeDrivers.includes(d.code);
                        const c = driverColors[d.code] || '#aaa';
                        return (
                            <button
                                key={d.code}
                                onClick={(e) => {
                                    if (e.shiftKey) {
                                        useSessionStore.getState().setSoloDriver(soloDriver === d.code ? null : d.code);
                                    } else {
                                        useSessionStore.getState().toggleDriver(d.code);
                                    }
                                }}
                                style={{
                                    fontSize: 10,
                                    fontFamily: 'JetBrains Mono',
                                    fontWeight: isActive ? 800 : 500,
                                    padding: '2px 6px',
                                    borderRadius: 4,
                                    border: `1px solid ${isActive ? c : 'rgba(255,255,255,0.06)'}`,
                                    backgroundColor: isActive ? `${c}20` : 'transparent',
                                    color: isActive ? '#fff' : '#6a6a8c',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s ease',
                                }}
                                title={`${d.name} - ${d.team}`}
                            >
                                {d.code}
                            </button>
                        );
                    })}
                </div>
            </div>
            <div className="panel-body flush" style={{ flex: 1, minHeight: 0 }}>
                <ReactECharts
                    ref={chartRef}
                    option={option}
                    style={{ height: '100%', minHeight: 180, width: '100%' }}
                    opts={{ renderer: 'canvas' }}
                />
            </div>
        </div>
    );
}
