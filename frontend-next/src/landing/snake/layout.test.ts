import { describe, expect, it } from 'vitest';
import type { LandingLayout, ToneBand } from '../tone/tone';
import { bezierPoint, lengthAtDepth, type Pt } from './geometry';
import { SNAKE, buildRail, buildSnake, gutterCenter } from './layout';

const BANDS: ToneBand[] = [
  { id: 'hero', tone: 'light', top: 0, bottom: 2600 },
  { id: 'pipeline', tone: 'dark', top: 2600, bottom: 5840 }, // the pinned track at vh = 900
  { id: 'models', tone: 'light', top: 5840, bottom: 6940 },
  { id: 'limits', tone: 'dark', top: 6940, bottom: 7940 },
  { id: 'closing', tone: 'light', top: 7940, bottom: 8640 },
  { id: 'site-footer', tone: 'dark', top: 8640, bottom: 9340 },
];
const DESKTOP: LandingLayout = { bands: BANDS, vw: 1440, vh: 900, height: 9340 };
const MOBILE: LandingLayout = { bands: BANDS, vw: 390, vh: 844, height: 9340 };

const pointsBetween = (pts: Pt[], y0: number, y1: number) => pts.filter((p) => p.y >= y0 && p.y <= y1);

const curls = (pts: Pt[]) => {
  for (let i = 0; i < pts.length - 1; i += 1)
    for (let j = i + 2; j < pts.length - 1; j += 1) {
      const [a, b, c, d] = [pts[i], pts[i + 1], pts[j], pts[j + 1]];
      const den = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);
      if (den === 0) continue;
      const u = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / den;
      const v = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) / den;
      if (u > 0 && u < 1 && v > 0 && v < 1) return true;
    }
  return false;
};

describe('buildSnake (desktop)', () => {
  const geo = buildSnake(DESKTOP)!;
  const samples: Pt[] = geo.segments.flatMap((s) => Array.from({ length: 30 }, (_, i) => bezierPoint(s, i / 30)));

  it('returns null without the snake bands', () => {
    expect(buildSnake({ ...DESKTOP, bands: BANDS.slice(0, 1) })).toBeNull();
  });

  it('starts 24px into the pipeline band and ends where the head can reach at max scroll', () => {
    expect(geo.lut.y[0]).toBe(2600 + SNAKE.startInset);
    const end = geo.lut.y[geo.lut.y.length - 1];
    expect(end).toBeLessThanOrEqual(Math.min(8640 - SNAKE.endInset, 9340 - 900 * (1 - SNAKE.headLine) - 4));
    const maxScroll = DESKTOP.height - DESKTOP.vh;
    expect(lengthAtDepth(geo.lut, maxScroll + SNAKE.headLine * DESKTOP.vh)).toBe(geo.lut.total);
  });

  it('runs the full pinned track in the left gutter', () => {
    const g = gutterCenter(1440);
    const onTrack = pointsBetween(samples, 2600 + SNAKE.startInset, 5840 - SNAKE.crossHalf - 1);
    expect(onTrack.length).toBeGreaterThan(100);
    expect(onTrack.every((p) => p.x <= 2 * g)).toBe(true);
    expect(Math.max(...onTrack.map((p) => p.y))).toBeGreaterThan(5840 - SNAKE.crossHalf - 20);
  });

  it('stays inside the viewport width', () => {
    expect(Array.from(geo.lut.x).every((x) => x >= 0 && x <= 1440)).toBe(true);
  });

  it('curls before the results and the closing band, not before the limits', () => {
    expect(curls(pointsBetween(samples, 5840 - 100, 5840 + 100))).toBe(true);
    expect(curls(pointsBetween(samples, 6940 - 100, 6940 + 100))).toBe(false);
    expect(curls(pointsBetween(samples, 7940 - 100, 7940 + 100))).toBe(true);
  });

  it('is deterministic', () => {
    expect(buildSnake(DESKTOP)!.d).toBe(geo.d);
  });
});

describe('buildSnake (mobile)', () => {
  const geo = buildSnake(MOBILE)!;

  it('is one rail inside the right padding', () => {
    expect(geo.mobile).toBe(true);
    expect(Array.from(geo.lut.x).every((x) => x >= 390 - 20 && x <= 390)).toBe(true);
  });

  it('has no curls: the frontier never jumps', () => {
    let prev = lengthAtDepth(geo.lut, 2700);
    let worst = 0;
    for (let Y = 2701; Y < 8500; Y += 1) {
      const L = lengthAtDepth(geo.lut, Y);
      worst = Math.max(worst, L - prev);
      prev = L;
    }
    expect(worst).toBeLessThan(5);
  });
});

describe('buildRail (inner pages)', () => {
  const layout: LandingLayout = { bands: BANDS, vw: 1440, vh: 900, height: 9340 };
  const geo = buildRail(layout)!;
  it('is one rail on the right gutter with no crossings', () => {
    const xs = geo.segments.flatMap((s) => [s[0].x, s[3].x]);
    expect(Math.min(...xs)).toBeGreaterThan(1440 / 2);
  });
  it('stops above the footer and within reach of the head', () => {
    expect(geo.lut.y[geo.lut.y.length - 1]).toBeLessThanOrEqual(8640 - SNAKE.endInset);
  });
});
