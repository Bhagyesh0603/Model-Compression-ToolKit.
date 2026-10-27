'use client';

import { useEffect, useRef } from 'react';
import { animate, useInView, useMotionValueEvent, type MotionValue } from 'motion/react';
import { usePrefersReducedMotion } from '@/src/landing/hooks';
import { EASE } from './tokens';

/**
 * A number that follows a MotionValue without re-rendering React: the text node
 * is written directly each frame. Server output is the formatted `initial`.
 */
export function MotionNumber({
  value,
  format,
  initial,
  className,
}: {
  value: MotionValue<number>;
  format: (v: number) => string;
  initial?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useMotionValueEvent(value, 'change', (v) => {
    if (ref.current) ref.current.textContent = format(v);
  });
  return (
    <span ref={ref} className={`figures-tab ${className ?? ''}`}>
      {format(initial ?? value.get())}
    </span>
  );
}

/** Counts from `from` to `to` the first time it is on screen. At rest it shows exactly `to`. */
export function CountUp({
  to,
  from = 0,
  format,
  duration = 1.2,
  className,
}: {
  to: number;
  from?: number;
  format: (v: number) => string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true, margin: '0px 0px -10% 0px' });
  const reduce = usePrefersReducedMotion();
  useEffect(() => {
    if (!seen || reduce) return;
    const el = ref.current;
    const controls = animate(from, to, {
      duration,
      ease: EASE,
      onUpdate: (v) => {
        if (el) el.textContent = format(v);
      },
    });
    return () => controls.stop();
  }, [seen, reduce, from, to, duration, format]);
  return (
    <span ref={ref} className={`figures-tab ${className ?? ''}`}>
      {format(to)}
    </span>
  );
}

