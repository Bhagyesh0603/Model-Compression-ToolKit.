type Props = {
  indices: number[];
  original: number[];
  recalibrated: number[];
  className?: string;
};

const W = 520;
const H = 190;
const PAD = { l: 44, r: 12, t: 14, b: 30 };

/**
 * Running variance of one BatchNorm layer, per kept channel. Hollow dot: the
 * statistic the network was trained with (and still carries right after the
 * cut). Solid dot: re-estimated on the pruned network. The line joins the two.
 */
export function BnShift({ indices, original, recalibrated, className }: Props) {
  const kept = indices.map((i) => original[i]);
  const all = [...kept, ...recalibrated];
  const hi = Math.max(...all) * 1.1;
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const x = (k: number) => PAD.l + ((k + 0.5) / indices.length) * iw;
  const y = (v: number) => PAD.t + (1 - v / hi) * ih;
  const ticks = [0, hi / 2, hi].map((t) => Math.round(t * 100) / 100);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="img" aria-label="Running variance per kept channel, before and after re-estimation">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={0.12} />
          <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.55} className="figures-tab">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {indices.map((idx, k) => (
        <g key={idx}>
          <line x1={x(k)} x2={x(k)} y1={y(kept[k])} y2={y(recalibrated[k])} stroke="currentColor" strokeOpacity={0.4} />
          <circle cx={x(k)} cy={y(kept[k])} r={4} fill="var(--bg, #fff)" stroke="currentColor" strokeWidth={1.5} />
          <circle cx={x(k)} cy={y(recalibrated[k])} r={4} fill="var(--color-brand-600)" />
          <text x={x(k)} y={H - 10} textAnchor="middle" fontSize={10} fill="currentColor" fillOpacity={0.6}>
            {idx}
          </text>
        </g>
      ))}
      <text x={PAD.l} y={H - 0} fontSize={9.5} fill="currentColor" fillOpacity={0.5}>
        original channel index
      </text>
    </svg>
  );
}
