import type { PruneStep } from '@/src/data/results';

type Props = {
  baseline: number;
  history: Pick<PruneStep, 'step' | 'acc_after_cut' | 'acc_after_finetune'>[];
  className?: string;
  /** Plain-language description for screen readers. */
  summary: string;
  tone?: 'page' | 'glass';
};

const W = 360;
const H = 168;
const PAD = { l: 38, r: 10, t: 12, b: 26 };

/**
 * Accuracy through a gradual prune: each step drops accuracy (open dot, right
 * after the cut) and fine-tuning recovers part of it (solid dot). Both values
 * come straight from a run's `prune_history`.
 */
export function Sawtooth({ baseline, history, className, summary, tone = 'page' }: Props) {
  const lows = history.map((h) => h.acc_after_cut);
  const lo = Math.floor((Math.min(...lows) - 0.02) * 20) / 20;
  const hi = Math.ceil((baseline + 0.005) * 20) / 20;
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const slots = history.length * 2;
  const x = (slot: number) => PAD.l + (slot / slots) * iw;
  const y = (acc: number) => PAD.t + (1 - (acc - lo) / (hi - lo)) * ih;

  const points: { slot: number; acc: number; kind: 'base' | 'cut' | 'tuned'; step?: number }[] = [
    { slot: 0, acc: baseline, kind: 'base' },
  ];
  history.forEach((h, i) => {
    points.push({ slot: 2 * i + 1, acc: h.acc_after_cut, kind: 'cut', step: h.step });
    points.push({ slot: 2 * i + 2, acc: h.acc_after_finetune, kind: 'tuned', step: h.step });
  });
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.slot).toFixed(1)} ${y(p.acc).toFixed(1)}`).join(' ');

  const ticks: number[] = [];
  const tickStep = hi - lo > 0.3 ? 0.1 : 0.05;
  for (let t = Math.ceil(lo / tickStep - 1e-9) * tickStep; t <= hi + 1e-9; t += tickStep) ticks.push(Math.round(t * 100) / 100);
  const ink = tone === 'glass' ? '#141414' : 'currentColor';

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="img" aria-label={summary}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke={ink} strokeOpacity={0.12} strokeWidth={1} />
          <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill={ink} fillOpacity={0.55} className="figures-tab">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke={ink} strokeOpacity={0.55} strokeWidth={1.4} strokeLinejoin="round" />
      {points.map((p) => (
        <circle
          key={p.slot}
          cx={x(p.slot)}
          cy={y(p.acc)}
          r={p.kind === 'base' ? 4 : 3.6}
          fill={p.kind === 'cut' ? (tone === 'glass' ? '#fff' : 'var(--bg, #fff)') : 'var(--color-brand-600)'}
          stroke={p.kind === 'cut' ? ink : 'var(--color-brand-600)'}
          strokeWidth={1.6}
        />
      ))}
      <text x={x(0)} y={H - 8} fontSize={10} fill={ink} fillOpacity={0.6} textAnchor="middle">
        base
      </text>
      {history.map((h, i) => (
        <text key={h.step} x={x(2 * i + 1.5)} y={H - 8} fontSize={10} fill={ink} fillOpacity={0.6} textAnchor="middle">
          {h.step}
        </text>
      ))}
    </svg>
  );
}
