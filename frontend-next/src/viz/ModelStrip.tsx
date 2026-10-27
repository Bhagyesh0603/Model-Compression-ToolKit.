'use client';

import { useId } from 'react';
import { motion, useTransform, type MotionValue } from 'motion/react';
import type { StripLayer } from './layers';

const CELL = 2; // one channel = 2 units of height
const COL = 16;
const GAP = 6;

/**
 * A model drawn as columns of channels, one column per pruned layer. The pale
 * column is the original width; the solid part is what the pruner kept; the
 * amber wash is the same channels once quantized to INT8. `cut` and `quant`
 * are 0..1 motion values, so a scroll, a timeline or a slider can drive it
 * without re-rendering. Channel counts come from the run's recorded prune stats.
 */
export function ModelStrip({
  layers,
  cut,
  quant,
  className,
  label,
}: {
  layers: StripLayer[];
  cut: MotionValue<number>;
  quant: MotionValue<number>;
  className?: string;
  label: string;
}) {
  const uid = useId().replace(/:/g, '');
  const maxTotal = Math.max(...layers.map((l) => l.total));
  const H = maxTotal * CELL;
  const W = layers.length * (COL + GAP) - GAP;
  const pat = (suffix: string) => `${uid}-${suffix}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="img" aria-label={label} preserveAspectRatio="xMidYMax meet">
      <defs>
        {(
          [
            ['base', 'var(--line-strong)', 0.7],
            ['keep', 'var(--snake)', 1],
            ['int8', 'var(--color-int8)', 1],
          ] as const
        ).map(([k, fill, o]) => (
          <pattern key={k} id={pat(k)} width={COL} height={CELL} patternUnits="userSpaceOnUse">
            <rect width={COL} height={CELL - 0.55} fill={fill} fillOpacity={o} />
          </pattern>
        ))}
      </defs>
      {layers.map((l, i) => (
        <Column key={l.name} layer={l} x={i * (COL + GAP)} H={H} cut={cut} quant={quant} pat={pat} />
      ))}
    </svg>
  );
}

function Column({
  layer,
  x,
  H,
  cut,
  quant,
  pat,
}: {
  layer: StripLayer;
  x: number;
  H: number;
  cut: MotionValue<number>;
  quant: MotionValue<number>;
  pat: (s: string) => string;
}) {
  const fullH = layer.total * CELL;
  const keptCells = useTransform(cut, (c) => layer.total - c * (layer.total - layer.kept));
  const height = useTransform(keptCells, (k) => k * CELL);
  const y = useTransform(height, (h) => H - h);
  return (
    <g>
      <rect x={x} y={H - fullH} width={COL} height={fullH} fill={`url(#${pat('base')})`} />
      <motion.rect x={x} y={y} width={COL} height={height} fill={`url(#${pat('keep')})`} />
      <motion.rect x={x} y={y} width={COL} height={height} fill={`url(#${pat('int8')})`} style={{ opacity: quant }} />
    </g>
  );
}
