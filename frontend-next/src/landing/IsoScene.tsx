import { useId } from 'react';

/**
 * Isometric block field — clay-white towers, an evergreen core and a few
 * glass boxes with green cubes inside. Generated once from a fixed seed, so
 * server and client render the same markup and nothing ships as an image.
 */

type Kind = 'white' | 'brand' | 'glass';
type Cell = { i: number; j: number; z: number; kind: Kind; float: boolean };

const HW = 64; // half tile width
const HH = 32; // half tile height
const U = 56; // screen units per height unit
const VIEW = { x: -760, y: -480, w: 1520, h: 960 };

function rand(i: number, j: number, k: number): number {
  let a = (i * 374761393 + j * 668265263 + k * 1274126177 + 0x9e3779b9) | 0;
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const q = (n: number) => Math.round(n * 4) / 4;

function cellAt(i: number, j: number): Cell | null {
  const r = Math.hypot(i + 0.5, j + 0.5);
  const roll = rand(i, j, 1);
  const h = rand(i, j, 2);

  if (r < 2.4) {
    return { i, j, z: q(0.3 + h * 0.45), kind: 'brand', float: false };
  }
  if (r < 4.4) {
    if (roll < 0.34) return null;
    if (roll < 0.54) return { i, j, z: q(1 + h * 0.7), kind: 'glass', float: rand(i, j, 3) < 0.6 };
    return { i, j, z: q(0.35 + h * 0.8), kind: 'white', float: false };
  }
  if (roll < 0.05) return null;
  if (roll < 0.12) return { i, j, z: q(1 + h * 1.2), kind: 'glass', float: rand(i, j, 3) < 0.5 };
  const z = q(Math.min(5, 0.8 + (r - 4.4) * 0.34 + h * 1.5));
  if (roll < 0.22) return { i, j, z: q(z * 0.6), kind: 'brand', float: false };
  return { i, j, z, kind: 'white', float: false };
}

function visible(c: Cell): boolean {
  const x0 = (c.i - c.j - 1) * HW;
  const x1 = (c.i - c.j + 1) * HW;
  const y0 = (c.i + c.j) * HH - c.z * U - 40;
  const y1 = (c.i + c.j + 2) * HH;
  return x1 > VIEW.x && x0 < VIEW.x + VIEW.w && y1 > VIEW.y && y0 < VIEW.y + VIEW.h;
}

const CELLS: Cell[] = (() => {
  const out: Cell[] = [];
  for (let i = -22; i <= 22; i++) {
    for (let j = -22; j <= 22; j++) {
      const c = cellAt(i, j);
      if (c && visible(c)) out.push(c);
    }
  }
  return out.sort((a, b) => a.i + a.j - (b.i + b.j) || a.i - b.i);
})();

const r1 = (n: number) => Math.round(n * 10) / 10;
const pt = (x: number, y: number, z: number) => `${r1((x - y) * HW)},${r1((x + y) * HH - z * U)}`;

function box(x0: number, y0: number, x1: number, y1: number, z0: number, z1: number) {
  return {
    top: `M${pt(x0, y0, z1)}L${pt(x1, y0, z1)}L${pt(x1, y1, z1)}L${pt(x0, y1, z1)}Z`,
    right: `M${pt(x1, y0, z1)}L${pt(x1, y1, z1)}L${pt(x1, y1, z0)}L${pt(x1, y0, z0)}Z`,
    left: `M${pt(x0, y1, z1)}L${pt(x1, y1, z1)}L${pt(x1, y1, z0)}L${pt(x0, y1, z0)}Z`,
    hidden: `M${pt(x0, y0, z0)}L${pt(x0, y0, z1)}M${pt(x0, y0, z0)}L${pt(x1, y0, z0)}M${pt(x0, y0, z0)}L${pt(x0, y1, z0)}`,
  };
}

type Props = {
  className?: string;
};

export function IsoScene({ className }: Props) {
  const id = useId().replace(/:/g, '');
  const g = (name: string) => `url(#${id}-${name})`;

  return (
    <svg
      viewBox={`${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`}
      preserveAspectRatio="xMidYMid slice"
      className={className}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`${id}-wTop`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#eeefed" />
        </linearGradient>
        <linearGradient id={`${id}-wLeft`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f3f4f2" />
          <stop offset="1" stopColor="#dadbd8" />
        </linearGradient>
        <linearGradient id={`${id}-wRight`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#dcdddb" />
          <stop offset="1" stopColor="#c1c3bf" />
        </linearGradient>
        <linearGradient id={`${id}-oTop`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" style={{ stopColor: 'var(--color-brand-200)' }} />
          <stop offset="1" style={{ stopColor: 'var(--color-brand-300)' }} />
        </linearGradient>
        <linearGradient id={`${id}-oLeft`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" style={{ stopColor: 'var(--color-brand-400)' }} />
          <stop offset="1" style={{ stopColor: 'var(--color-brand-500)' }} />
        </linearGradient>
        <linearGradient id={`${id}-oRight`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" style={{ stopColor: 'var(--color-brand-500)' }} />
          <stop offset="1" style={{ stopColor: 'var(--color-brand-700)' }} />
        </linearGradient>
        <radialGradient id={`${id}-glow`}>
          <stop offset="0" style={{ stopColor: 'var(--color-brand-300)', stopOpacity: 0.75 }} />
          <stop offset="1" style={{ stopColor: 'var(--color-brand-300)', stopOpacity: 0 }} />
        </radialGradient>
      </defs>

      <rect x={VIEW.x} y={VIEW.y} width={VIEW.w} height={VIEW.h} fill="#e8e9e7" />
      <ellipse cx="0" cy="40" rx="520" ry="300" fill={g('glow')} />

      {CELLS.map((c) => {
        const key = `${c.i}:${c.j}`;
        if (c.kind === 'glass') {
          const f = box(c.i, c.j, c.i + 1, c.j + 1, 0, c.z);
          const inner = box(c.i + 0.3, c.j + 0.3, c.i + 0.7, c.j + 0.7, 0, Math.min(0.45, c.z * 0.4));
          return (
            <g key={key}>
              <path d={f.hidden} fill="none" stroke="var(--color-brand-400)" strokeOpacity="0.35" strokeWidth="1" />
              <g
                className={c.float ? 'ss-float' : undefined}
                style={
                  c.float
                    ? ({
                        '--ss-float-distance': '-18px',
                        '--ss-float-duration': `${5 + rand(c.i, c.j, 4) * 3}s`,
                        '--ss-float-delay': `${-rand(c.i, c.j, 5) * 4}s`,
                      } as React.CSSProperties)
                    : undefined
                }
              >
                <path d={inner.left} fill={g('oLeft')} />
                <path d={inner.right} fill={g('oRight')} />
                <path d={inner.top} fill={g('oTop')} />
              </g>
              <g fill="#ffffff" stroke="var(--color-brand-400)" strokeOpacity="0.85" strokeWidth="1.2" strokeLinejoin="round">
                <path d={f.left} fillOpacity="0.16" />
                <path d={f.right} fillOpacity="0.1" />
                <path d={f.top} fillOpacity="0.28" />
              </g>
            </g>
          );
        }
        const f = box(c.i, c.j, c.i + 1, c.j + 1, 0, c.z);
        const o = c.kind === 'brand';
        return (
          <g key={key}>
            <path d={f.left} fill={g(o ? 'oLeft' : 'wLeft')} />
            <path d={f.right} fill={g(o ? 'oRight' : 'wRight')} />
            <path d={f.top} fill={g(o ? 'oTop' : 'wTop')} stroke={o ? 'var(--color-brand-200)' : '#ffffff'} strokeWidth="1.2" />
          </g>
        );
      })}
    </svg>
  );
}
