import { describe, expect, it } from 'vitest';
import {
  FOLLOW,
  bezierPoint,
  buildLut,
  crossSegments,
  lengthAtDepth,
  pointAtLength,
  railSegments,
  stepFollow,
  toPathD,
  type Cubic,
  type Follow,
  type Pt,
} from './geometry';

const sample = (segs: readonly Cubic[], per = 40): Pt[] => {
  const pts: Pt[] = [];
  for (const s of segs) for (let i = 0; i < per; i += 1) pts.push(bezierPoint(s, i / per));
  if (segs.length) pts.push(segs[segs.length - 1][3]);
  return pts;
};

const segmentsCross = (a: Pt, b: Pt, c: Pt, d: Pt) => {
  const den = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);
  if (den === 0) return false;
  const u = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / den;
  const v = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) / den;
  return u > 0 && u < 1 && v > 0 && v < 1;
};

const selfIntersects = (pts: Pt[]) => {
  for (let i = 0; i < pts.length - 1; i += 1)
    for (let j = i + 2; j < pts.length - 1; j += 1) if (segmentsCross(pts[i], pts[i + 1], pts[j], pts[j + 1])) return true;
  return false;
};

/** Every join shares its point and its tangent direction (G1). */
const smooth = (segs: readonly Cubic[]) =>
  segs.every((s, i) => {
    if (i === 0) return true;
    const p = segs[i - 1];
    if (Math.hypot(s[0].x - p[3].x, s[0].y - p[3].y) > 1e-9) return false;
    const t1 = { x: p[3].x - p[2].x, y: p[3].y - p[2].y };
    const t2 = { x: s[1].x - s[0].x, y: s[1].y - s[0].y };
    const cross = (t1.x * t2.y - t1.y * t2.x) / (Math.hypot(t1.x, t1.y) * Math.hypot(t2.x, t2.y));
    return Math.abs(cross) < 1e-6 && t1.x * t2.x + t1.y * t2.y > 0;
  });

const neverClimbs = (pts: Pt[]) => pts.every((p, i) => i === 0 || p.y >= pts[i - 1].y - 1e-9);

describe('railSegments', () => {
  const rail = railSegments(100, 0, 1000, 20, 360);

  it('runs from (x, y0) to (x, y1)', () => {
    expect(rail[0][0]).toEqual({ x: 100, y: 0 });
    expect(rail[rail.length - 1][3]).toEqual({ x: 100, y: 1000 });
  });

  it('stays within ±amplitude, never climbs, and joins smoothly', () => {
    const pts = sample(rail);
    expect(pts.every((p) => p.x >= 80 - 1e-9 && p.x <= 120 + 1e-9)).toBe(true);
    expect(neverClimbs(pts)).toBe(true);
    expect(smooth(rail)).toBe(true);
  });

  it('is empty for an empty range', () => {
    expect(railSegments(0, 10, 10, 4, 100)).toEqual([]);
  });
});

describe('crossSegments', () => {
  const from = { x: 40, y: 0 };
  const to = { x: 1400, y: 192 };

  it('leaves and arrives heading straight down', () => {
    const c = crossSegments(from, to, 96);
    expect(c[0][1].x).toBe(from.x);
    expect(c[c.length - 1][2].x).toBe(to.x);
  });

  it('a plain crossing never climbs', () => {
    expect(neverClimbs(sample(crossSegments(from, to, 96), 60))).toBe(true);
  });

  it('a looped crossing curls (self-intersects) and stays smooth', () => {
    const c = crossSegments(from, to, 96, 44);
    expect(selfIntersects(sample(c, 30))).toBe(true);
    expect(smooth(c)).toBe(true);
    expect(c[0][0]).toEqual(from);
    expect(c[c.length - 1][3]).toEqual(to);
  });
});

describe('toPathD', () => {
  it('is deterministic and starts with a move', () => {
    const segs = railSegments(100, 0, 1000, 20, 360);
    expect(toPathD(segs)).toBe(toPathD(railSegments(100, 0, 1000, 20, 360)));
    expect(toPathD(segs).startsWith('M100 0C')).toBe(true);
    expect(toPathD([])).toBe('');
  });
});

describe('lookup table and frontier', () => {
  const path: Cubic[] = [
    ...railSegments(40, 0, 900, 20, 360),
    ...crossSegments({ x: 40, y: 900 }, { x: 1400, y: 1092 }, 996, 44),
    ...railSegments(1400, 1092, 2000, 20, 360),
  ];
  const lut = buildLut(path);

  it('measures a straight cubic exactly', () => {
    const line: Cubic = [{ x: 0, y: 0 }, { x: 0, y: 100 }, { x: 0, y: 200 }, { x: 0, y: 300 }];
    expect(buildLut([line]).total).toBeCloseTo(300, 2);
  });

  it('puts the head exactly on the requested depth, loops included', () => {
    for (let Y = 1; Y < 2000; Y += 0.5) {
      const p = pointAtLength(lut, lengthAtDepth(lut, Y));
      expect(Math.abs(p.y - Y)).toBeLessThan(0.01);
    }
  });

  it('is monotone in depth and clamps outside the path', () => {
    let prev = -1;
    for (let Y = 0; Y < 2100; Y += 1) {
      const L = lengthAtDepth(lut, Y);
      expect(L).toBeGreaterThanOrEqual(prev);
      prev = L;
    }
    expect(lengthAtDepth(lut, -5)).toBe(0);
    expect(lengthAtDepth(lut, 5000)).toBe(lut.total);
  });

  it('jumps (by more than FOLLOW.jumpPx) only around the crossing', () => {
    const jumps: number[] = [];
    let prev = lengthAtDepth(lut, 0);
    for (let Y = 0.5; Y < 2000; Y += 0.5) {
      const L = lengthAtDepth(lut, Y);
      if (L - prev > FOLLOW.jumpPx) jumps.push(Y);
      prev = L;
    }
    expect(jumps.length).toBeGreaterThan(0);
    expect(jumps.every((Y) => Y > 976 && Y < 1016)).toBe(true);
  });
});

describe('stepFollow', () => {
  it('tracks small gaps exactly', () => {
    expect(stepFollow({ drawn: 0, catching: false }, 50, 16)).toEqual({ drawn: 50, catching: false });
  });

  it('eases across a large gap without overshoot and settles within a second', () => {
    let f: Follow = stepFollow({ drawn: 0, catching: false }, 500, 16);
    expect(f.catching).toBe(true);
    expect(f.drawn).toBeGreaterThan(0);
    expect(f.drawn).toBeLessThan(500);
    let frames = 0;
    while (f.catching && frames < 200) {
      f = stepFollow(f, 500, 16);
      expect(f.drawn).toBeLessThanOrEqual(500);
      frames += 1;
    }
    expect(f).toEqual({ drawn: 500, catching: false });
    expect(frames).toBeLessThan(60);
  });
});
