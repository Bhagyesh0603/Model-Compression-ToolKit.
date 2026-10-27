/**
 * One motion language for every page. These are the values the SafeSpeak site
 * already used, named once so a chart, a pin and a reveal all move alike.
 */
export const EASE = [0.2, 0.7, 0.2, 1] as const;
export const DUR = { micro: 0.3, enter: 0.5, layout: 0.7, reveal: 0.9 } as const;
export const STAGGER = 0.055;

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
/** Progress of `p` through the window [a, b], clamped to 0..1. */
export const windowed = (p: number, a: number, b: number) => clamp01((p - a) / (b - a));
