
/**
 * Landing tone model — pure, no DOM.
 *
 * The page is a stack of bands (sections) that each declare a tone. A single
 * horizontal switch line (60% down the viewport) decides which band is active;
 * hysteresis stops the choice flapping at a boundary, and a scroll floor
 * guarantees the incoming tone is fully applied before a pinned section pins
 * (its top reaches the viewport top).
 */

export type Tone = 'light' | 'dark';

export interface ToneBand {
  /** Element id of the band, e.g. "how". */
  id: string;
  tone: Tone;
  /** Top edge in landing-root coordinates (px). */
  top: number;
  /** Bottom edge in landing-root coordinates: the next band's top, or the root height. */
  bottom: number;
}

/** Everything measured from the DOM in one pass (LandingController.measure). */
export interface LandingLayout {
  bands: readonly ToneBand[];
  /** Viewport width without the scrollbar (documentElement.clientWidth). */
  vw: number;
  vh: number;
  /** Height of the .landing root (px). */
  height: number;
}

/** How the active band was entered; decides which edge the scroll floor tracks. */
export type Entry = 'above' | 'below' | 'jump';

export interface Selection {
  index: number;
  entry: Entry;
}

export const TONE = {
  /** The switch line, as a fraction of the viewport height from its top. */
  switchLine: 0.6,
  /** A neighbouring band only takes over once the line is this far past the shared edge. */
  hysteresisPx: 48,
  /** Scrolling down: complete by the time the band's top reaches 15% of the viewport (a pin happens at 0%). */
  completeDown: 0.15,
  /** Scrolling up: complete by the time the band's bottom edge reaches 95% (a pinned panel fills the screen at 100%). */
  completeUp: 0.95,
  /** A programmatic scroll with more than this many viewports left selects by its destination. */
  jumpVh: 1.5,
  /** Per-frame lerp of the colour mix, about half Lenis's: the light/dark fade takes ~1s so it reads as a fade, not a flash. */
  lerp: 0.05,
  /** A mix closer than this to its target snaps and counts as settled. */
  epsilon: 0.004,
} as const;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/**
 * Page-space y of the switch line. Over the last viewport of scroll the line
 * slides from 60% to 100% of the viewport, so the final band (the footer) is
 * always reachable even when it is shorter than 40% of the viewport.
 */
export function lineY(scrollY: number, vh: number, maxScroll: number): number {
  const remaining = Math.max(0, maxScroll - scrollY);
  const t = vh > 0 ? clamp01(1 - remaining / vh) : 1;
  return scrollY + vh * (TONE.switchLine + (1 - TONE.switchLine) * t);
}

/** Index of the band containing y (bands sorted by top). Clamps outside the range; -1 when empty. */
export function bandIndexAt(bands: readonly ToneBand[], y: number): number {
  if (bands.length === 0) return -1;
  if (y < bands[0].top) return 0;
  let lo = 0;
  let hi = bands.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (bands[mid].top <= y) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Chooses the active band for a switch line at y. Moving to a neighbour needs
 * the line to be hysteresisPx past the shared edge; anything further away
 * (anchor jump, resize, first frame) is taken immediately as a 'jump'.
 */
export function selectBand(bands: readonly ToneBand[], y: number, current: Selection | null): Selection {
  const candidate = bandIndexAt(bands, y);
  if (!current || current.index < 0 || current.index >= bands.length) return { index: candidate, entry: 'jump' };
  const i = current.index;
  if (candidate === i) return current;
  if (candidate === i + 1) {
    return y >= bands[i + 1].top + TONE.hysteresisPx ? { index: i + 1, entry: 'above' } : current;
  }
  if (candidate === i - 1) {
    return y < bands[i].top - TONE.hysteresisPx ? { index: i - 1, entry: 'below' } : current;
  }
  return { index: candidate, entry: 'jump' };
}

/**
 * Minimum transition progress (0..1) demanded by the scroll position alone.
 * `top` is the viewport top in root coordinates. 0 at the moment of switching,
 * 1 once the crossed edge reaches completeDown (or completeUp scrolling up).
 */
export function scrollFloor(bands: readonly ToneBand[], sel: Selection, top: number, vh: number): number {
  const band = bands[sel.index];
  if (!band || sel.entry === 'jump') return 0;
  if (sel.entry === 'above') {
    const edge = band.top - top;
    const start = vh * TONE.switchLine - TONE.hysteresisPx;
    const end = vh * TONE.completeDown;
    return start > end ? clamp01((start - edge) / (start - end)) : 1;
  }
  const edge = band.bottom - top;
  const start = vh * TONE.switchLine + TONE.hysteresisPx;
  const end = vh * TONE.completeUp;
  return end > start ? clamp01((edge - start) / (end - start)) : 1;
}

/**
 * Advances the dark mix (0 = light, 1 = dark) toward `target` with Lenis's
 * per-frame lerp, never lagging the scroll floor. `instant` (reduced motion,
 * first frame) jumps straight to the target.
 */
export function stepMix(mix: number, target: 0 | 1, dtMs: number, floor: number, instant: boolean): number {
  if (instant) return target;
  const frames = Math.max(0, dtMs) / (1000 / 60);
  let next = target + (mix - target) * Math.pow(1 - TONE.lerp, frames);
  next = target === 1 ? Math.max(next, floor) : Math.min(next, 1 - floor);
  return Math.abs(next - target) < TONE.epsilon ? target : next;
}
