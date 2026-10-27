import type { StageMeasure } from '@/src/data/results';

const W = 520;
const H = 240;
const PAD = { l: 48, r: 18, t: 18, b: 40 };

/** Accuracy against measured median latency for the stages of one model, joined in pipeline order. */
export function ParetoMini({ stages, className }: { stages: StageMeasure[]; className?: string }) {
  const xs = stages.map((s) => s.median_ms);
  const ys = stages.map((s) => s.accuracy * 100);
  const xHi = Math.max(...xs) * 1.08;
  const xLo = 0;
  const yLo = Math.floor(Math.min(...ys) - 2);
  const yHi = Math.ceil(Math.max(...ys) + 1);
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const x = (v: number) => PAD.l + ((v - xLo) / (xHi - xLo)) * iw;
  const y = (v: number) => PAD.t + (1 - (v - yLo) / (yHi - yLo)) * ih;
  const ticks = [yLo, (yLo + yHi) / 2, yHi].map((t) => Math.round(t));
  const path = stages.map((s, i) => `${i ? 'L' : 'M'}${x(s.median_ms).toFixed(1)} ${y(s.accuracy * 100).toFixed(1)}`).join(' ');
  const maxSize = Math.max(...stages.map((s) => s.size_mb));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="img" aria-label="Accuracy against median latency for each stage">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={0.12} />
          <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.55} className="figures-tab">
            {t}%
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke="currentColor" strokeOpacity={0.35} strokeWidth={1.3} strokeDasharray="4 4" />
      {stages.map((s, i) => (
        <g key={s.stage}>
          <circle
            cx={x(s.median_ms)}
            cy={y(s.accuracy * 100)}
            r={5 + 9 * Math.sqrt(s.size_mb / maxSize)}
            fill={s.precision === 'int8' ? 'var(--color-brand-600)' : 'currentColor'}
            fillOpacity={s.precision === 'int8' ? 0.8 : 0.22}
          />
          <text
            x={x(s.median_ms)}
            y={
              // Neighbouring stages often land almost on top of each other; alternate above and below.
              i % 2 === 1
                ? y(s.accuracy * 100) + (18 + 9 * Math.sqrt(s.size_mb / maxSize))
                : y(s.accuracy * 100) - (12 + 9 * Math.sqrt(s.size_mb / maxSize))
            }
            textAnchor={i === 0 ? 'end' : 'middle'}
            fontSize={10}
            fill="currentColor"
            fillOpacity={0.75}
          >
            {s.stage}
          </text>
        </g>
      ))}
      <text x={(PAD.l + W - PAD.r) / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="currentColor" fillOpacity={0.6}>
        median latency (ms), marker area is file size
      </text>
      <text x={PAD.l} y={H - 24} fontSize={10} fill="currentColor" fillOpacity={0.55} className="figures-tab">
        0
      </text>
      <text x={W - PAD.r} y={H - 24} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.55} className="figures-tab">
        {xHi.toFixed(2)}
      </text>
    </svg>
  );
}
