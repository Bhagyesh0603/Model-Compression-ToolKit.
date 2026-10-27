/**
 * Snake geometry — pure, no DOM. Builds the path from cubic Bézier segments
 * and answers the one question the renderer asks every frame: how much of the
 * path is drawn when the head must sit at depth Y?
 */

export interface Pt {
  x: number;
  y: number;
}

/** A cubic Bézier: start, control 1, control 2, end. */
export type Cubic = readonly [Pt, Pt, Pt, Pt];

/** Arc-length lookup table sampled every ~LUT_STEP px of path. */
export interface Lut {
  /** Cumulative length at each sample (len[0] = 0). */
  len: Float64Array;
  x: Float64Array;
  y: Float64Array;
  /** Running maximum of y up to each sample: the deepest point drawn so far. */
  reach: Float64Array;
  total: number;
}

export const LUT_STEP = 2;

/** Curl shape as fractions of its radius: drift per radian (< 1 makes it cross itself), descent per radian, centre lift. */
export const LOOP_SHAPE = { drift: 0.42, descent: 0.16, lift: 0.3 } as const;

/** Drawn-length follower: exact up to jumpPx, eased (time constant tauMs) across bigger gaps such as a curl. */
export const FOLLOW = { jumpPx: 120, tauMs: 90, settlePx: 0.5 } as const;

const DOWN: Pt = { x: 0, y: 1 };

function unit(v: Pt): Pt {
  const m = Math.hypot(v.x, v.y) || 1;
  return { x: v.x / m, y: v.y / m };
}

/** Cubic from p0 leaving along unit d0 (handle h0) into p1 arriving along unit d1 (handle h1). */
export function hermite(p0: Pt, d0: Pt, h0: number, p1: Pt, d1: Pt, h1: number): Cubic {
  return [p0, { x: p0.x + d0.x * h0, y: p0.y + d0.y * h0 }, { x: p1.x - d1.x * h1, y: p1.y - d1.y * h1 }, p1];
}

export function bezierPoint(c: Cubic, t: number): Pt {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const d = 3 * u * t * t;
  const e = t * t * t;
  return {
    x: a * c[0].x + b * c[1].x + d * c[2].x + e * c[3].x,
    y: a * c[0].y + b * c[1].y + d * c[2].y + e * c[3].y,
  };
}

/**
 * A gentle vertical serpentine centred on x from y0 to y1. Every knot has a
 * vertical tangent (so rails and crossings join smoothly), and handles of
 * 0.45 × step keep y monotone.
 */
export function railSegments(x: number, y0: number, y1: number, amplitude: number, halfWave: number): Cubic[] {
  if (y1 <= y0) return [];
  const n = Math.max(1, Math.round((y1 - y0) / halfWave));
  const step = (y1 - y0) / n;
  const knotX = (k: number) => (k === 0 || k === n ? x : x + amplitude * (k % 2 === 1 ? 1 : -1));
  const h = 0.45 * step;
  const out: Cubic[] = [];
  for (let k = 0; k < n; k += 1) {
    out.push(hermite({ x: knotX(k), y: y0 + k * step }, DOWN, h, { x: knotX(k + 1), y: y0 + (k + 1) * step }, DOWN, h));
  }
  return out;
}

/**
 * One full curl centred on `center`, travelling horizontally in `dir` while
 * descending: the prolate cycloid x = cx + dir(a·t − r·sin t), y = cy − r·cos t + c·t
 * for t ∈ [−π, π], fitted with one Hermite cubic per step.
 */
export function loopSegments(center: Pt, r: number, dir: 1 | -1, steps = 12): Cubic[] {
  const a = LOOP_SHAPE.drift * r;
  const c = LOOP_SHAPE.descent * r;
  const cy = center.y - LOOP_SHAPE.lift * r;
  const at = (t: number): Pt => ({ x: center.x + dir * (a * t - r * Math.sin(t)), y: cy - r * Math.cos(t) + c * t });
  const slope = (t: number): Pt => ({ x: dir * (a - r * Math.cos(t)), y: r * Math.sin(t) + c });
  const dt = (2 * Math.PI) / steps;
  const out: Cubic[] = [];
  for (let i = 0; i < steps; i += 1) {
    const t0 = -Math.PI + i * dt;
    const t1 = t0 + dt;
    const p0 = at(t0);
    const p3 = at(t1);
    const s0 = slope(t0);
    const s1 = slope(t1);
    out.push([
      p0,
      { x: p0.x + (s0.x * dt) / 3, y: p0.y + (s0.y * dt) / 3 },
      { x: p3.x - (s1.x * dt) / 3, y: p3.y - (s1.y * dt) / 3 },
      p3,
    ]);
  }
  return out;
}

/**
 * Crosses from one gutter to the other around boundary y `midY`: leaves `from`
 * heading down, runs horizontally (optionally through a curl of radius
 * `loopRadius` at the page centre) and arrives at `to` heading down.
 */
export function crossSegments(from: Pt, to: Pt, midY: number, loopRadius = 0): Cubic[] {
  const dir: 1 | -1 = to.x >= from.x ? 1 : -1;
  const cx = (from.x + to.x) / 2;
  if (loopRadius <= 0) {
    const mid: Pt = { x: cx, y: Math.min(Math.max(midY, from.y + 8), to.y - 8) };
    const across: Pt = { x: dir, y: 0 };
    const half = Math.abs(cx - from.x);
    return [
      hermite(from, DOWN, 0.9 * (mid.y - from.y), mid, across, 0.5 * half),
      hermite(mid, across, 0.5 * half, to, DOWN, 0.9 * (to.y - mid.y)),
    ];
  }
  const loop = loopSegments({ x: cx, y: midY }, loopRadius, dir);
  const first = loop[0];
  const last = loop[loop.length - 1];
  const start = first[0];
  const end = last[3];
  const startDir = unit({ x: first[1].x - start.x, y: first[1].y - start.y });
  const endDir = unit({ x: end.x - last[2].x, y: end.y - last[2].y });
  return [
    hermite(from, DOWN, 0.9 * Math.max(8, start.y - from.y), start, startDir, 0.5 * Math.abs(start.x - from.x)),
    ...loop,
    hermite(end, endDir, 0.5 * Math.abs(to.x - end.x), to, DOWN, 0.9 * Math.max(8, to.y - end.y)),
  ];
}

const r1 = (v: number) => Math.round(v * 10) / 10;

/** SVG path data for a continuous chain of cubics. */
export function toPathD(segments: readonly Cubic[]): string {
  if (segments.length === 0) return '';
  const parts = [`M${r1(segments[0][0].x)} ${r1(segments[0][0].y)}`];
  for (const [, c1, c2, p] of segments) {
    parts.push(`C${r1(c1.x)} ${r1(c1.y)} ${r1(c2.x)} ${r1(c2.y)} ${r1(p.x)} ${r1(p.y)}`);
  }
  return parts.join('');
}

/** Samples the chain into an arc-length table. Built once per layout, never per frame. */
export function buildLut(segments: readonly Cubic[], step = LUT_STEP): Lut {
  const xs: number[] = [];
  const ys: number[] = [];
  const ls: number[] = [];
  let total = 0;
  if (segments.length > 0) {
    xs.push(segments[0][0].x);
    ys.push(segments[0][0].y);
    ls.push(0);
  }
  for (const seg of segments) {
    const hull =
      Math.hypot(seg[1].x - seg[0].x, seg[1].y - seg[0].y) +
      Math.hypot(seg[2].x - seg[1].x, seg[2].y - seg[1].y) +
      Math.hypot(seg[3].x - seg[2].x, seg[3].y - seg[2].y);
    const n = Math.max(2, Math.ceil(hull / step));
    let prev = seg[0];
    for (let i = 1; i <= n; i += 1) {
      const p = bezierPoint(seg, i / n);
      total += Math.hypot(p.x - prev.x, p.y - prev.y);
      xs.push(p.x);
      ys.push(p.y);
      ls.push(total);
      prev = p;
    }
  }
  const reach = new Float64Array(ys.length);
  let deepest = -Infinity;
  for (let i = 0; i < ys.length; i += 1) {
    deepest = Math.max(deepest, ys[i]);
    reach[i] = deepest;
  }
  return { len: Float64Array.from(ls), x: Float64Array.from(xs), y: Float64Array.from(ys), reach, total };
}

/**
 * The frontier: the shortest drawn length whose path has reached depth Y.
 * Because it is the FIRST point at depth Y, y(lengthAtDepth(Y)) === Y for every
 * Y inside the path's range — the head sits exactly on its line however many
 * loops came before. Monotone non-decreasing; it jumps across a curl's upper
 * arc (smoothed by stepFollow).
 */
export function lengthAtDepth(lut: Lut, Y: number): number {
  const n = lut.len.length;
  if (n === 0 || Y <= lut.y[0]) return 0;
  if (Y > lut.reach[n - 1]) return lut.total;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (lut.reach[mid] >= Y) hi = mid;
    else lo = mid + 1;
  }
  if (lo === 0) return 0;
  const y0 = lut.y[lo - 1];
  const y1 = lut.y[lo];
  const f = y1 > y0 ? (Y - y0) / (y1 - y0) : 1;
  return lut.len[lo - 1] + f * (lut.len[lo] - lut.len[lo - 1]);
}

/** Point at arc length s (clamped), interpolated between samples. */
export function pointAtLength(lut: Lut, s: number): Pt {
  const n = lut.len.length;
  if (n === 0) return { x: 0, y: 0 };
  if (s <= 0) return { x: lut.x[0], y: lut.y[0] };
  if (s >= lut.total) return { x: lut.x[n - 1], y: lut.y[n - 1] };
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (lut.len[mid] >= s) hi = mid;
    else lo = mid + 1;
  }
  const s0 = lut.len[lo - 1];
  const s1 = lut.len[lo];
  const f = s1 > s0 ? (s - s0) / (s1 - s0) : 0;
  return { x: lut.x[lo - 1] + f * (lut.x[lo] - lut.x[lo - 1]), y: lut.y[lo - 1] + f * (lut.y[lo] - lut.y[lo - 1]) };
}

export interface Follow {
  /** Drawn length (px). */
  drawn: number;
  /** True while easing across a gap larger than FOLLOW.jumpPx. */
  catching: boolean;
}

/**
 * The drawn length follows the frontier exactly, except across gaps larger
 * than FOLLOW.jumpPx (a curl's upper arc, a fast flick through a crossing),
 * which it covers with an exponential ease so the head visibly runs the curl.
 */
export function stepFollow(prev: Follow, target: number, dtMs: number): Follow {
  const gap = target - prev.drawn;
  if (!prev.catching && Math.abs(gap) <= FOLLOW.jumpPx) return { drawn: target, catching: false };
  const next = prev.drawn + gap * (1 - Math.exp(-Math.max(0, dtMs) / FOLLOW.tauMs));
  if (Math.abs(target - next) < FOLLOW.settlePx) return { drawn: target, catching: false };
  return { drawn: next, catching: true };
}
