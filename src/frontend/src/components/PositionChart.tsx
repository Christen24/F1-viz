/**
 * PositionChart — Bump chart showing driver position evolution across laps
 *
 * ECharts line chart with inverted Y-axis (P1 top), one line per driver,
 * progressive reveal synced with lap playback, hover/click interaction.
 * Upgraded to match Bento Grid glassmorphism styling.
 */
import ReactECharts from 'echarts-for-react';
import { useLapPlaybackStore } from '../stores/lapPlaybackStore';
import { useSessionStore } from '../stores/sessionStore';
import { useMemo, useCallback } from 'react';

export function PositionChart() {
    const lapData = useLapPlaybackStore(s => s.lapData);
    const currentLap = useLapPlaybackStore(s => s.currentLap);
    const totalLaps = useLapPlaybackStore(s => s.totalLaps);
    const meta = useSessionStore(s => s.metadata);
    const selectedDrivers = useSessionStore(s => s.selectedDrivers);

    const lapsUpTo = useMemo(() => lapData.slice(0, currentLap), [lapData, currentLap]);

    const driverColors = useMemo(() => {
        if (!meta) return {};
        const c: Record<string, string> = {};
        meta.drivers.forEach(d => { c[d.code] = d.team_color; });
        return c;
    }, [meta]);

    const allDrivers = useMemo(() => {
        if (!meta) return [];
        return meta.drivers.map(d => d.code);
    }, [meta]);

    const { series, overtakeMarkers } = useMemo(() => {
        if (!lapsUpTo.length || !allDrivers.length) return { series: [], overtakeMarkers: [] };

        const markers: { lap: number; pos: number; driver: string }[] = [];

        const s = allDrivers.map(code => {
            const data: [number, number | null][] = [];
            let prevPos: number | null = null;

            for (const lap of lapsUpTo) {
                const pos = lap.positions[code];
                if (pos !== undefined) {
                    data.push([lap.lap, pos]);
                    if (prevPos !== null && pos !== prevPos) {
                        markers.push({ lap: lap.lap, pos, driver: code });
                    }
                    prevPos = pos;
                } else {
                    data.push([lap.lap, null]);
                }
            }

            const isSelected = selectedDrivers.length === 0 || selectedDrivers.includes(code);
            const color = driverColors[code] || '#888';

            return {
                name: code,
                type: 'line' as const,
                data,
                smooth: 0.3,
                symbol: 'none',
                lineStyle: {
                    width: isSelected ? 2 : 1,
                    color,
                    opacity: isSelected ? 1 : 0.2,
                },
                itemStyle: { color },
                emphasis: {
                    lineStyle: { width: 3, opacity: 1 },
                    focus: 'series' as const,
                },
                blur: {
                    lineStyle: { opacity: 0.08, width: 1 },
                },
                z: isSelected ? 2 : 1,
                animationDuration: 400,
                animationEasing: 'cubicOut' as const,
            };
        });

        return { series: s, overtakeMarkers: markers };
    }, [lapsUpTo, allDrivers, driverColors, selectedDrivers]);

    const overtakeSeries = useMemo(() => {
        if (!overtakeMarkers.length) return null;
        return {
            name: '_overtakes',
            type: 'scatter' as const,
            data: overtakeMarkers.map(m => ({
                value: [m.lap, m.pos],
                itemStyle: { color: driverColors[m.driver] || '#ff6b35', borderColor: '#fff', borderWidth: 1 },
            })),
            symbolSize: 6,
            z: 10,
            silent: true,
            emphasis: { disabled: true },
            animation: false,
        };
    }, [overtakeMarkers, driverColors]);

    const handleClick = useCallback((params: { seriesName?: string }) => {
        if (params.seriesName && params.seriesName !== '_overtakes') {
            useSessionStore.getState().toggleDriver(params.seriesName);
        }
    }, []);

    if (!meta || !lapsUpTo.length) return null;

    const allSeries = overtakeSeries ? [...series, overtakeSeries] : series;

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis' as const,
            backgroundColor: 'rgba(7, 8, 11, 0.95)',
            borderColor: 'rgba(255, 255, 255, 0.15)',
            padding: [10, 14],
            borderRadius: 8,
            textStyle: { color: '#fff', fontSize: 11, fontFamily: 'Inter' },
            formatter: (params: Array<{ seriesName: string; data: [number, number] }>) => {
                if (!Array.isArray(params)) return '';
                const lap = params[0]?.data?.[0];
                if (!lap) return '';
                const prevLap = lapData[lap - 2];
                const lines = params
                    .filter(p => p.seriesName !== '_overtakes' && p.data?.[1] != null)
                    .sort((a, b) => (a.data[1] ?? 99) - (b.data[1] ?? 99))
                    .slice(0, 10)
                    .map(p => {
                        const pos = p.data[1];
                        const prevPos = prevLap?.positions[p.seriesName];
                        let change = '';
                        if (prevPos !== undefined && prevPos !== pos) {
                            const diff = prevPos - pos;
                            change = diff > 0
                                ? ` <span style="color:#00d27a">▲${diff}</span>`
                                : ` <span style="color:#e10600">▼${Math.abs(diff)}</span>`;
                        }
                        const color = driverColors[p.seriesName] || '#888';
                        return `<span style="color:${color}; font-weight:800">●</span> P${pos} ${p.seriesName}${change}`;
                    });
                return `<div style="margin-bottom: 4px; font-weight: 700; color: rgba(255,255,255,0.7)">Lap ${lap}</div>${lines.join('<br/>')}`;
            },
        },
        legend: { show: false },
        grid: { left: 35, right: 12, top: 15, bottom: 25 },
        xAxis: {
            type: 'value' as const,
            min: 1,
            max: totalLaps || 'dataMax',
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
            axisLabel: { color: '#9898b0', fontSize: 10, fontFamily: 'JetBrains Mono' },
            splitLine: { show: false },
        },
        yAxis: {
            type: 'value' as const,
            min: 1,
            max: 20,
            inverse: true,
            axisLine: { show: false },
            axisLabel: {
                color: '#9898b0', fontSize: 10, fontFamily: 'JetBrains Mono',
                formatter: (v: number) => `P${v}`,
            },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
            interval: 1,
        },
        ...(currentLap > 1 ? { markLine: undefined } : {}),
        dataZoom: [{ type: 'inside' as const, xAxisIndex: 0 }],
        series: [
            ...allSeries,
            {
                name: '_cursor',
                type: 'line' as const,
                markLine: {
                    silent: true,
                    symbol: 'none',
                    lineStyle: { color: '#e10600', width: 1, type: 'dashed' as const, opacity: 0.6 },
                    data: [{ xAxis: currentLap }],
                    label: { show: false },
                    animation: false,
                },
                data: [],
            },
        ],
    };

    return (
        <div className="card">
            <div className="panel-hdr">
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="panel-icon">📈</span> Position Evolution
                </span>
                <span style={{ fontSize: 10, color: 'var(--color-text-muted)' }}>
                    {selectedDrivers.length > 0 ? `${selectedDrivers.length} selected` : 'All drivers'}
                </span>
            </div>
            <div className="panel-body flush">
                <ReactECharts
                    option={option}
                    style={{ height: 260 }}
                    onEvents={{ click: handleClick }}
                    notMerge={false}
                    lazyUpdate={true}
                />
            </div>
        </div>
    );
}
