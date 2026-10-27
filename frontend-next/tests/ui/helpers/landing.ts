import type { Page } from '@playwright/test';
import { lineY } from '../../../src/landing/tone/tone';

export interface BandBox {
  top: number;
  height: number;
  vh: number;
  maxScroll: number;
}

/** Opens / with the intro curtain skipped and waits for the first band measurement. */
export async function openLanding(page: Page): Promise<void> {
  await page.addInitScript(() => {
    try {
      sessionStorage.setItem('compresskit.intro', '1');
    } catch {
      /* private mode */
    }
  });
  await page.goto('/', { waitUntil: 'load' });
  await page.waitForFunction(() => Number(document.querySelector<HTMLElement>('.landing')?.dataset.measureCount ?? 0) > 0);
  await waitLayoutSettled(page);
}

/**
 * Waits for fonts and the load event, then for the controller's re-measures
 * (ResizeObserver, fonts.ready, load, 120 ms debounce) to go quiet, so a late
 * re-measure cannot land in the middle of a test.
 */
export async function waitLayoutSettled(page: Page, quietMs = 400): Promise<void> {
  await page.evaluate(
    (quiet) =>
      new Promise<void>((resolve) => {
        const count = () => document.querySelector<HTMLElement>('.landing')?.dataset.measureCount ?? '0';
        const start = () => {
          let last = count();
          let since = performance.now();
          const poll = () => {
            const now = count();
            if (now !== last) {
              last = now;
              since = performance.now();
            }
            if (performance.now() - since >= quiet) resolve();
            else setTimeout(poll, 50);
          };
          poll();
        };
        const fonts = document.fonts ? document.fonts.ready.then(() => undefined) : Promise.resolve();
        const loaded =
          document.readyState === 'complete' ? Promise.resolve() : new Promise<void>((r) => window.addEventListener('load', () => r(), { once: true }));
        void Promise.all([fonts, loaded]).then(start);
      }),
    quietMs,
  );
}

/** Resolves after `count` animation frames. */
export async function nextFrames(page: Page, count = 2): Promise<void> {
  await page.evaluate(
    (n) =>
      new Promise<void>((resolve) => {
        let left = n;
        const tick = () => {
          left -= 1;
          if (left <= 0) resolve();
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
    count,
  );
}

export async function scrollToY(page: Page, y: number): Promise<void> {
  await page.evaluate((top) => window.scrollTo(0, top), Math.max(0, Math.round(y)));
  await nextFrames(page, 2);
}

export async function waitToneSettled(page: Page): Promise<void> {
  await page.waitForFunction(() => document.querySelector<HTMLElement>('.landing')?.dataset.toneSettled === 'true');
}

export async function bandBox(page: Page, id: string): Promise<BandBox> {
  return page.evaluate((bandId) => {
    const el = document.getElementById(bandId);
    if (!el) throw new Error(`band #${bandId} not found`);
    const r = el.getBoundingClientRect();
    return {
      top: r.top + window.scrollY,
      height: r.height,
      vh: window.innerHeight,
      maxScroll: document.documentElement.scrollHeight - window.innerHeight,
    };
  }, id);
}

/** scrollY at which the switch line (same formula as the app) sits at page-y `target`. */
export function scrollForLine(box: BandBox, target: number): number {
  let lo = 0;
  let hi = box.maxScroll;
  for (let i = 0; i < 40; i += 1) {
    const mid = (lo + hi) / 2;
    if (lineY(mid, box.vh, box.maxScroll) < target) lo = mid;
    else hi = mid;
  }
  return hi;
}

/** Background of the .landing root as sRGB bytes (color-mix values are parsed by a canvas). */
export async function rootRgb(page: Page): Promise<[number, number, number]> {
  return page.evaluate(() => {
    const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
    ctx.fillStyle = '#000';
    ctx.fillStyle = getComputedStyle(document.querySelector('.landing')!).backgroundColor;
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
    return [r, g, b] as [number, number, number];
  });
}

export const near = (a: readonly number[], b: readonly number[], tolerance = 2) =>
  a.every((v, i) => Math.abs(v - b[i]) <= tolerance);

/**
 * Text inside a band whose contrast against its nearest opaque background is
 * below WCAG AA (4.5:1, or 3:1 for >= 24px / >= 18.66px bold). Skips hidden,
 * aria-hidden, sr-only and [data-tone-fixed] content and anything over a
 * gradient/image background.
 */
export async function contrastOffenders(page: Page, bandId: string): Promise<string[]> {
  return page.evaluate((id) => {
    const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
    const rgba = (value: string): [number, number, number, number] | null => {
      if (!value || value === 'transparent' || value === 'none') return null;
      ctx.clearRect(0, 0, 1, 1);
      ctx.fillStyle = '#000';
      ctx.fillStyle = value;
      ctx.fillRect(0, 0, 1, 1);
      const d = ctx.getImageData(0, 0, 1, 1).data;
      return [d[0], d[1], d[2], d[3] / 255];
    };
    const lum = (c: number[]) => {
      const f = (v: number) => {
        const s = v / 255;
        return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
    };
    const ratio = (a: number[], b: number[]) => {
      const [hi, lo] = [lum(a), lum(b)].sort((m, n) => n - m);
      return (hi + 0.05) / (lo + 0.05);
    };
    const band = document.getElementById(id);
    if (!band) return [`band #${id} missing`];
    const out: string[] = [];
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    for (const el of Array.from(band.querySelectorAll<HTMLElement>('h1,h2,h3,p,a,button,li,dt,dd,span'))) {
      if (el.closest('[aria-hidden="true"], .sr-only, [data-tone-fixed], [hidden]')) continue;
      const text = Array.from(el.childNodes)
        .filter((n) => n.nodeType === Node.TEXT_NODE)
        .map((n) => n.textContent ?? '')
        .join('')
        .trim();
      if (!text) continue;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0 || r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) continue;
      const cs = getComputedStyle(el);
      const fg = rgba(cs.color);
      if (!fg || fg[3] < 0.1) continue;
      let bg: number[] | null = null;
      for (let node: HTMLElement | null = el; node; node = node.parentElement) {
        const s = getComputedStyle(node);
        if (s.backgroundImage !== 'none') break;
        const c = rgba(s.backgroundColor);
        if (c && c[3] > 0.95) {
          bg = c;
          break;
        }
      }
      if (!bg) continue;
      const mixed = [0, 1, 2].map((i) => fg[i] * fg[3] + bg![i] * (1 - fg[3]));
      const size = parseFloat(cs.fontSize);
      const min = size >= 24 || (size >= 18.66 && Number(cs.fontWeight) >= 700) ? 3 : 4.5;
      const cr = ratio(mixed, bg);
      if (cr < min) out.push(`${el.tagName.toLowerCase()} "${text.slice(0, 40)}" ${cr.toFixed(2)} < ${min}`);
    }
    return out;
  }, bandId);
}
