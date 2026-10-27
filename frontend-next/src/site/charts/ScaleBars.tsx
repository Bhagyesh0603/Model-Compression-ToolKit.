/** One INT8 scale per output channel: a channel with small weights gets a finer grid. */
export function ScaleBars({ scales, className }: { scales: number[]; className?: string }) {
  const max = Math.max(...scales);
  const W = 560;
  const H = 120;
  const bw = W / scales.length;
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} className={className} role="img" aria-label="INT8 scale of each output channel">
      {scales.map((s, i) => (
        <rect key={i} x={i * bw + 1.5} y={H - (s / max) * H} width={bw - 3} height={(s / max) * H} rx={2} fill="var(--color-brand-600)" fillOpacity={0.8} />
      ))}
      <text x={0} y={H + 14} fontSize={10} fill="currentColor" fillOpacity={0.6}>
        output channel 0
      </text>
      <text x={W} y={H + 14} fontSize={10} textAnchor="end" fill="currentColor" fillOpacity={0.6}>
        {scales.length - 1}
      </text>
    </svg>
  );
}
