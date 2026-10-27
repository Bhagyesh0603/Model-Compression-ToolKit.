'use client';

import { motion, useTransform, type MotionValue } from 'motion/react';
import type { PruneStep } from '@/src/data/results';

const W = 760;
const H = 230;
const PAD = { l: 48, r: 16, t: 14, b: 30 };

type Pt = { acc: number; kind: 'base' | 'cut' | 'tuned'; step?: number };

/**
 * Accuracy through a gradual prune, drawn as `reveal` (0..1) advances: the line
 * runs left to right and each dot appears as the line reaches it. Open dots are
 * accuracy right after a cut, solid dots after the fine-tune that follows.
 * Every value is one recorded `prune_history` row.
 */
export function AnimSawtooth({
  baseline,
  history,
  reveal,
  summary,
  className,
}: {
  baseline: number;
  history: Pick<PruneStep, 'step' | 'acc_after_cut' | 'acc_after_finetune'>[];
  reveal: MotionValue<number>;
  summary: string;
  className?: string;
}) {
  const pts: Pt[] = [{ acc: baseline, kind: 'base' }];
  history.forEach((h) => {
    pts.push({ acc: h.acc_after_cut, kind: 'cut', step: h.step });
    pts.push({ acc: h.acc_after_finetune, kind: 'tuned', step: h.step });
  });
  const lo = Math.floor((Math.min(...pts.map((p) => p.acc)) - 0.02) * 20) / 20;
  const hi = Math.ceil((baseline + 0.005) * 20) / 20;
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (i / (pts.length - 1)) * iw;
  const y = (a: number) => PAD.t + (1 - (a - lo) / (hi - lo)) * ih;
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(p.acc).toFixed(1)}`).join(' ');
  const tickStep = hi - lo > 0.3 ? 0.1 : 0.05;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / tickStep - 1e-9) * tickStep; t <= hi + 1e-9; t += tickStep) ticks.push(Math.round(t * 100) / 100);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="img" aria-label={summary}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
          <text x={PAD.l - 7} y={y(t) + 3.5} textAnchor="end" fontSize={12} fill="var(--muted)" className="figures-tab">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      <motion.path d={d} fill="none" stroke="var(--fg)" strokeOpacity={0.6} strokeWidth={1.5} strokeLinejoin="round" style={{ pathLength: reveal }} />
      {pts.map((p, i) => (
        <Dot key={i} cx={x(i)} cy={y(p.acc)} kind={p.kind} at={i / (pts.length - 1)} reveal={reveal} />
      ))}
      <text x={x(0)} y={H - 8} fontSize={12} fill="var(--muted)" textAnchor="middle">
        base
      </text>
      {history.map((h, i) => (
        <text key={h.step} x={(x(2 * i + 1) + x(2 * i + 2)) / 2} y={H - 8} fontSize={12} fill="var(--muted)" textAnchor="middle">
          cut {h.step}
        </text>
      ))}
    </svg>
  );
}

function Dot({ cx, cy, kind, at, reveal }: { cx: number; cy: number; kind: Pt['kind']; at: number; reveal: MotionValue<number> }) {
  const opacity = useTransform(reveal, [Math.max(0, at - 0.03), Math.max(0.001, at)], [0, 1]);
  return (
    <motion.circle
      cx={cx}
      cy={cy}
      r={kind === 'base' ? 5 : 4.6}
      fill={kind === 'cut' ? 'var(--bg)' : 'var(--snake)'}
      stroke={kind === 'cut' ? 'var(--fg)' : 'var(--snake)'}
      strokeWidth={1.6}
      style={{ opacity }}
    />
  );
}
