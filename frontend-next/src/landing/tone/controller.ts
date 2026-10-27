import type Lenis from 'lenis';
import {
  TONE,
  lineY,
  scrollFloor,
  selectBand,
  stepMix,
  type LandingLayout,
  type Selection,
  type Tone,
  type ToneBand,
} from './tone';

export interface LandingFrame {
  /** Viewport top in landing-root coordinates (window.scrollY − root page offset). */
  top: number;
  vw: number;
  vh: number;
  /** Milliseconds since the previous frame, clamped to [0, 34]. */
  dt: number;
}

/** Return true to request another frame (e.g. while easing). */
export type FrameListener = (frame: LandingFrame) => boolean;
export type LayoutListener = (layout: LandingLayout) => void;

export interface ToneSnapshot {
  tone: Tone;
  bandId: string | null;
}

const MAX_DT_MS = 34;

/**
 * Where Lenis's current smooth scroll will end. For a programmatic scrollTo
 * (anchor links, lenis.scrollTo) Lenis updates `targetScroll` to the in-flight
 * value every frame, so it is NOT the destination; the destination lives on the
 * (type-private) tween as `animate.to`. Wheel scrolling sets `targetScroll` to
 * the destination directly, which equals `animate.to`. Falls back to
 * `targetScroll` if a future Lenis hides the tween.
 */
function lenisDestination(lenis: Lenis): number {
  const tween = (lenis as unknown as { animate?: { isRunning?: boolean; to?: number } }).animate;
  return tween?.isRunning && typeof tween.to === 'number' ? tween.to : lenis.targetScroll;
}
const MEASURE_DEBOUNCE_MS = 120;

/**
 * The landing page's single scroll loop.
 *
 * One passive scroll listener schedules at most one requestAnimationFrame. A
 * frame reads only window.scrollY plus cached geometry, writes one CSS number
 * (--tone-dark) and two data attributes, then drives subscribers (the snake).
 * Layout is read only in measure(): on attach, window resize, root resize
 * (ResizeObserver), font load and window load — debounced, never per frame.
 */
export class LandingController {
  private root: HTMLElement | null = null;
  private lenis: Lenis | null = null;
  private reduced = false;
  private layout: LandingLayout | null = null;
  private rootTop = 0;
  private maxScroll = 0;
  private selection: Selection | null = null;
  private mix = 0;
  private settled = true;
  private firstFrame = true;
  private lock: number | null = null;
  private raf = 0;
  private lastFrame = 0;
  private measureTimer = 0;
  private measureCount = 0;
  private snapshot: ToneSnapshot = { tone: 'light', bandId: null };
  private readonly toneListeners = new Set<() => void>();
  private readonly frameListeners = new Set<FrameListener>();
  private readonly layoutListeners = new Set<LayoutListener>();
  private teardown: (() => void) | null = null;

  attach(root: HTMLElement): void {
    this.detach();
    this.root = root;
    this.firstFrame = true;
    const onScroll = () => this.request();
    const onResize = () => this.scheduleMeasure();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);
    window.addEventListener('load', onResize);
    const observer = new ResizeObserver(onResize);
    observer.observe(root);
    let alive = true;
    document.fonts?.ready
      .then(() => {
        if (alive) this.scheduleMeasure();
      })
      .catch(() => undefined);
    this.teardown = () => {
      alive = false;
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('load', onResize);
      observer.disconnect();
      window.clearTimeout(this.measureTimer);
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.lastFrame = 0;
    };
    this.measure();
  }

  detach(): void {
    this.teardown?.();
    this.teardown = null;
    this.root = null;
    this.selection = null;
    this.lock = null;
  }

  setLenis(lenis: Lenis | null): void {
    this.lenis = lenis;
  }

  setReducedMotion(reduced: boolean): void {
    this.reduced = reduced;
    this.request();
  }

  readonly subscribeTone = (listener: () => void): (() => void) => {
    this.toneListeners.add(listener);
    return () => {
      this.toneListeners.delete(listener);
    };
  };

  readonly getToneSnapshot = (): ToneSnapshot => this.snapshot;

  onFrame(listener: FrameListener): () => void {
    this.frameListeners.add(listener);
    this.request();
    return () => {
      this.frameListeners.delete(listener);
    };
  }

  onLayout(listener: LayoutListener): () => void {
    this.layoutListeners.add(listener);
    const current = this.layout;
    if (current) {
      queueMicrotask(() => {
        if (this.layoutListeners.has(listener)) listener(current);
      });
    }
    return () => {
      this.layoutListeners.delete(listener);
    };
  }

  private scheduleMeasure(): void {
    window.clearTimeout(this.measureTimer);
    this.measureTimer = window.setTimeout(() => this.measure(), MEASURE_DEBOUNCE_MS);
  }

  /** The only place that reads layout. Bands are measured on their non-sticky elements. */
  private measure(): void {
    const root = this.root;
    if (!root) return;
    const scrollY = window.scrollY;
    this.rootTop = root.getBoundingClientRect().top + scrollY;
    const height = root.offsetHeight;
    const found = Array.from(root.querySelectorAll<HTMLElement>('[data-tone-band]')).map((el, i) => ({
      id: el.id || `band-${i}`,
      tone: (el.dataset.tone === 'dark' ? 'dark' : 'light') as Tone,
      top: el.getBoundingClientRect().top + scrollY - this.rootTop,
    }));
    found.sort((a, b) => a.top - b.top);
    const bands: ToneBand[] = found.map((b, i) => ({ ...b, bottom: i + 1 < found.length ? found[i + 1].top : height }));
    const vh = window.innerHeight;
    const previous = this.layout;
    this.layout = { bands, vw: document.documentElement.clientWidth, vh, height };
    this.maxScroll = Math.max(0, document.documentElement.scrollHeight - vh);
    // Keep the active band (and how it was entered) across a re-measure: fonts, load and
    // ResizeObserver re-measure without moving anything, and dropping the selection would
    // lose the hysteresis and the pin floor. A different band set re-selects from scratch;
    // a real resize that moves the line far away is still taken as a jump by selectBand.
    const current = this.selection;
    const kept = previous && current ? previous.bands[current.index] : undefined;
    const index = previous && kept && bands.length === previous.bands.length ? bands.findIndex((b) => b.id === kept.id) : -1;
    this.selection = current && index >= 0 ? { index, entry: current.entry } : null;
    this.measureCount += 1;
    root.dataset.measureCount = String(this.measureCount);
    for (const listener of this.layoutListeners) listener(this.layout);
    this.request();
  }

  private request(): void {
    if (this.raf || !this.root) return;
    this.raf = requestAnimationFrame(this.frame);
  }

  private readonly frame = (now: number): void => {
    this.raf = 0;
    const root = this.root;
    const layout = this.layout;
    if (!root || !layout || layout.bands.length === 0) return;
    const dt = this.lastFrame ? Math.min(MAX_DT_MS, Math.max(0, now - this.lastFrame)) : 1000 / 60;
    this.lastFrame = now;
    const scrollY = window.scrollY;
    const { vh } = layout;

    // Programmatic jumps (anchor links, lenis.scrollTo): select by the destination
    // so the bands passed on the way never flash.
    const lenis = this.lenis;
    const destination = lenis ? lenisDestination(lenis) : 0;
    if (lenis && lenis.isScrolling === 'smooth' && Math.abs(destination - lenis.animatedScroll) > TONE.jumpVh * vh) {
      this.lock = destination;
    } else if (this.lock !== null && (!lenis || !lenis.isScrolling || Math.abs(destination - this.lock) > 1)) {
      this.lock = null;
    }
    const locked = this.lock !== null;

    const y = lineY(this.lock ?? scrollY, vh, this.maxScroll) - this.rootTop;
    const sel = selectBand(layout.bands, y, locked ? null : this.selection);
    this.selection = sel;
    const band = layout.bands[sel.index];
    const target = band.tone === 'dark' ? 1 : 0;
    const floor = locked ? 0 : scrollFloor(layout.bands, sel, scrollY - this.rootTop, vh);
    const mix = stepMix(this.mix, target, dt, floor, this.reduced || this.firstFrame);
    this.firstFrame = false;
    if (mix !== this.mix) {
      this.mix = mix;
      root.style.setProperty('--tone-dark', mix.toFixed(4));
    }
    const settled = mix === target;
    if (settled !== this.settled) {
      this.settled = settled;
      root.dataset.toneSettled = String(settled);
    }
    if (band.tone !== this.snapshot.tone || band.id !== this.snapshot.bandId) {
      root.dataset.tone = band.tone;
      this.snapshot = { tone: band.tone, bandId: band.id };
      for (const listener of this.toneListeners) listener();
    }

    let again = !settled || locked;
    const info: LandingFrame = { top: scrollY - this.rootTop, vw: layout.vw, vh, dt };
    for (const listener of this.frameListeners) if (listener(info)) again = true;
    if (again) this.request();
    else this.lastFrame = 0;
  };
}
