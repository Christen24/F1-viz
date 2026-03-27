/**
 * ScrollContainer — Reusable horizontal scroll wrapper
 *
 * Features:
 *   - Drag-to-scroll (desktop)
 *   - Mouse wheel → horizontal scroll
 *   - Scroll-snap
 *   - Gradient edge indicators
 *   - Auto-scroll active item to center
 *   - Keyboard arrow navigation
 *   - Touch swipe (native)
 *   - ARIA tablist role
 */
import { useRef, useState, useEffect, useCallback, type ReactNode } from 'react';

interface ScrollContainerProps {
    children: ReactNode;
    className?: string;
    /** Auto-scroll this index into view */
    activeIndex?: number;
    ariaLabel?: string;
}

export function ScrollContainer({ children, className = '', activeIndex, ariaLabel }: ScrollContainerProps) {
    const ref = useRef<HTMLDivElement>(null);
    const [canScrollLeft, setCanScrollLeft] = useState(false);
    const [canScrollRight, setCanScrollRight] = useState(false);
    const dragState = useRef({ isDragging: false, startX: 0, scrollLeft: 0 });

    // Update gradient indicators
    const updateScrollState = useCallback(() => {
        const el = ref.current;
        if (!el) return;
        setCanScrollLeft(el.scrollLeft > 2);
        setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 2);
    }, []);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        updateScrollState();
        el.addEventListener('scroll', updateScrollState, { passive: true });
        const ro = new ResizeObserver(updateScrollState);
        ro.observe(el);
        return () => {
            el.removeEventListener('scroll', updateScrollState);
            ro.disconnect();
        };
    }, [updateScrollState]);

    // Auto-scroll active item into center
    useEffect(() => {
        if (activeIndex === undefined || !ref.current) return;
        const child = ref.current.children[activeIndex] as HTMLElement;
        if (child) {
            child.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
        }
    }, [activeIndex]);

    // Mouse wheel → horizontal scroll
    const handleWheel = useCallback((e: React.WheelEvent) => {
        const el = ref.current;
        if (!el) return;
        if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            e.preventDefault();
            el.scrollLeft += e.deltaY;
        }
    }, []);

    // Drag-to-scroll
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        const el = ref.current;
        if (!el) return;
        dragState.current = { isDragging: true, startX: e.pageX - el.offsetLeft, scrollLeft: el.scrollLeft };
        el.style.cursor = 'grabbing';
        el.style.userSelect = 'none';
    }, []);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!dragState.current.isDragging || !ref.current) return;
        e.preventDefault();
        const x = e.pageX - ref.current.offsetLeft;
        const walk = (x - dragState.current.startX) * 1.5;
        ref.current.scrollLeft = dragState.current.scrollLeft - walk;
    }, []);

    const handleMouseUp = useCallback(() => {
        dragState.current.isDragging = false;
        if (ref.current) {
            ref.current.style.cursor = 'grab';
            ref.current.style.userSelect = '';
        }
    }, []);

    // Keyboard navigation
    const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
        const el = ref.current;
        if (!el) return;
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            el.scrollBy({ left: -80, behavior: 'smooth' });
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            el.scrollBy({ left: 80, behavior: 'smooth' });
        }
    }, []);

    return (
        <div className={`scroll-wrap ${className}`}>
            {canScrollLeft && <div className="scroll-fade scroll-fade-left" />}
            <div
                ref={ref}
                className="scroll-inner"
                role="tablist"
                aria-label={ariaLabel || 'Scrollable tabs'}
                tabIndex={0}
                onWheel={handleWheel}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onKeyDown={handleKeyDown}
            >
                {children}
            </div>
            {canScrollRight && <div className="scroll-fade scroll-fade-right" />}
        </div>
    );
}
