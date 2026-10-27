type Props = { weights: number[]; scale: number; className?: string };

const W = 560;
const H = 210;
const PAD = { l: 46, r: 12, t: 12, b: 30 };

/**
 * One output channel's weights against their INT8 rounding error. With a
 * per-channel scale every weight snaps to the nearest of 255 steps, so the
 * error (in units of one step) can never leave the band between -0.5 and +0.5.
 */
export function RoundingScatter({ weights, scale, className }: Props) {
  const lo = Math.min(...weights);
  const hi = Math.max(...weights);
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const x = (w: number) => PAD.l + ((w - lo) / (hi - lo)) * iw;
  const y = (e: number) => PAD.t + (0.5 - e) * ih;
  const points = weights.map((w) => {
    const q = w / scale;
    return { w, e: q - Math.round(q) };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="img" aria-label="INT8 rounding error of each weight in one channel, bounded by half a step">
      <rect x={PAD.l} y={y(0.5)} width={iw} height={ih} fill="var(--color-brand-600)" fillOpacity={0.07} />
      {[0.5, 0, -0.5].map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={t === 0 ? 0.3 : 0.15} strokeDasharray={t === 0 ? undefined : '4 4'} />
          <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.55} className="figures-tab">
            {t > 0 ? '+' : ''}
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {points.map((p, i) => (
        <circle key={i} cx={x(p.w)} cy={y(p.e)} r={2.6} fill="var(--color-brand-600)" fillOpacity={0.75} />
      ))}
      <text x={PAD.l} y={H - 8} fontSize={10} fill="currentColor" fillOpacity={0.6} className="figures-tab">
        {lo.toFixed(3)}
      </text>
      <text x={W - PAD.r} y={H - 8} fontSize={10} textAnchor="end" fill="currentColor" fillOpacity={0.6} className="figures-tab">
        {hi.toFixed(3)}
      </text>
      <text x={(PAD.l + W - PAD.r) / 2} y={H - 8} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        FP32 weight value
      </text>
    </svg>
  );
}
