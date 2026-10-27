'use client';

import { useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { quantWeights as q } from '@/src/data/results';
import { fakeQuantize } from '@/src/lib/derive';
import { DUR, EASE } from '@/src/motion/tokens';

const W = 760;
const H = 150;
const PAD = 24;

/**
 * Bit-width slider over the recorded weights of one output channel. Each weight
 * sits on the top row at its FP32 value and drops to the nearest level of the
 * integer grid on the bottom row. The 8-bit scale equals the recorded one.
 */
export function BitsExplorer() {
  const [bits, setBits] = useState(8);
  const w = q.channel0.weights;
  const maxAbs = useMemo(() => Math.max(...w.map(Math.abs)), [w]);
  const r = useMemo(() => fakeQuantize(w, bits), [w, bits]);
  const x = (v: number) => PAD + ((v + maxAbs) / (2 * maxAbs)) * (W - 2 * PAD);
  const grid = bits <= 5 ? Array.from({ length: 2 * r.qmax + 1 }, (_, i) => (i - r.qmax) * r.scale) : [];
  const maxErr = Math.max(...w.map((v, i) => Math.abs(v - r.deq[i])));

  return (
    <div>
      <div className="flex items-center gap-4">
        <label htmlFor="bits" className="label-xs !text-(--muted)">
          Bit width
        </label>
        <input
          id="bits"
          type="range"
          min={2}
          max={8}
          step={1}
          value={bits}
          onChange={(e) => setBits(Number(e.target.value))}
          className="w-full max-w-[320px] accent-[var(--snake)]"
        />
        <output className="figures-tab w-16 text-[15px]">{bits}-bit</output>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="mt-6 w-full" role="img" aria-label={`${w.length} weights of ${q.layer}, output channel 0, snapped to a ${bits}-bit grid.`}>
        {grid.map((g, i) => (
          <line key={i} x1={x(g)} x2={x(g)} y1={30} y2={H - 30} stroke="var(--line-strong)" strokeWidth={1} strokeDasharray="2 3" />
        ))}
        <text x={PAD} y={16} fontSize={11} fill="var(--muted)">FP32 value</text>
        <text x={PAD} y={H - 8} fontSize={11} fill="var(--muted)">{bits}-bit value</text>
        {w.map((v, i) => (
          <g key={i}>
            <motion.line
              x1={x(v)}
              y1={30}
              y2={H - 30}
              initial={false}
              animate={{ x2: x(r.deq[i]) }}
              transition={{ duration: DUR.layout, ease: EASE }}
              stroke="var(--line-strong)"
              strokeWidth={0.7}
            />
            <circle cx={x(v)} cy={30} r={2.2} fill="var(--snake)" />
            <motion.circle cy={H - 30} r={2.2} fill="var(--color-int8)" initial={false} animate={{ cx: x(r.deq[i]) }} transition={{ duration: DUR.layout, ease: EASE }} />
          </g>
        ))}
      </svg>

      <dl className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-2xl bg-(--line) sm:grid-cols-4">
        <Cell label="Distinct levels used" value={`${r.levels} of ${2 * r.qmax + 1}`} />
        <Cell label="Step size (scale)" value={r.scale.toFixed(5)} />
        <Cell label="Mean squared error" value={r.mse.toExponential(2)} />
        <Cell label="Largest error" value={maxErr.toFixed(5)} />
      </dl>
      <p className="mt-4 text-[13px] leading-[1.6] text-(--muted)">
        Computed here from the recorded weights of {q.layer}, output channel 0, with symmetric per-channel quantization
        (scale = largest weight / {r.qmax}). At 8 bits the scale is the recorded {q.channel0.scale.toFixed(5)}. Real INT8
        also quantizes activations; this shows weights only.
      </p>
    </div>
  );
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-(--bg) p-4">
      <dt className="label-xs !text-(--muted)">{label}</dt>
      <dd className="figures-tab mt-1.5 text-[18px] tracking-[-0.02em]">{value}</dd>
    </div>
  );
}
