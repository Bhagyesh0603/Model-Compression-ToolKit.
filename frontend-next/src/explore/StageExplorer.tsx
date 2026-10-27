'use client';

import { useEffect, useState } from 'react';
import { animate, useMotionValue } from 'motion/react';
import { count, mb, models, ms, pct, ratio, slugOf, type ModelRecord } from '@/src/data/results';
import { STAGE_LABELS } from '@/src/lib/budget';
import { MotionNumber } from '@/src/motion/MotionNumber';
import { DUR, EASE } from '@/src/motion/tokens';
import { ModelStrip } from '@/src/viz/ModelStrip';
import { layersOf } from '@/src/viz/layers';
import { usePrefersReducedMotion } from '@/src/landing/hooks';
import Link from 'next/link';

const fmtCount = (v: number) => count(Math.round(v));
const fmtPct = (v: number) => pct(v);
const fmtMb = (v: number) => mb(v);
const fmtMs = (v: number) => ms(v);

/** Pick a recorded model and a stage; every number is that stage's recorded measurement. */
export function StageExplorer({ fixedArch }: { fixedArch?: string } = {}) {
  const [mi, setMi] = useState(() => Math.max(0, models.findIndex((m) => m.arch === fixedArch)));
  const [stage, setStage] = useState(0);
  const model = models[mi];
  const rows = [model.baseline, ...model.stages];
  const row = rows[stage];
  const layers = layersOf(model);
  const reduce = usePrefersReducedMotion();

  const acc = useMotionValue(row.accuracy);
  const size = useMotionValue(row.size_mb);
  const lat = useMotionValue(row.median_ms);
  const par = useMotionValue(row.parameters);
  const cut = useMotionValue(stage >= 1 ? 1 : 0);
  const quant = useMotionValue(stage >= 3 ? 1 : 0);

  useEffect(() => {
    const to = { duration: reduce ? 0 : DUR.layout, ease: EASE };
    const controls = [
      animate(acc, row.accuracy, to),
      animate(size, row.size_mb, to),
      animate(lat, row.median_ms, to),
      animate(par, row.parameters, to),
      animate(cut, stage >= 1 ? 1 : 0, to),
      animate(quant, stage >= 3 ? 1 : 0, to),
    ];
    return () => controls.forEach((c) => c.stop());
  }, [row, stage, reduce, acc, size, lat, par, cut, quant]);

  const b = model.baseline;
  const note = model.provenance.note ?? (model.provenance.kind === 'transcribed' ? 'Typed in from the run log.' : '');

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)] lg:gap-14">
      <div className="space-y-8">
        {fixedArch ? null : (
        <fieldset>
          <legend className="label-xs !text-(--muted)">Model</legend>
          <div className="mt-3 grid gap-2">
            {models.map((m, i) => (
              <label
                key={m.arch}
                className="flex cursor-pointer items-center justify-between gap-3 rounded-2xl border border-(--line) px-4 py-3 text-[15px] transition-colors has-[:checked]:border-transparent has-[:checked]:bg-(--inv-bg) has-[:checked]:text-(--inv-fg) has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-brand-600 hover:bg-(--hover) has-[:checked]:hover:bg-(--inv-bg)"
              >
                <span>
                  <input type="radio" name="model" checked={mi === i} onChange={() => setMi(i)} className="sr-only" />
                  {m.name}
                  <span className="mt-0.5 block text-[12px] opacity-70">
                    {m.provenance.kind} · {m.provenance.host}
                  </span>
                </span>
                <span className={`h-2 w-2 rounded-full ${m.provenance.kind === 'measured' ? 'bg-(--snake)' : 'bg-int8'}`} aria-hidden="true" />
              </label>
            ))}
          </div>
        </fieldset>
        )}

        <div>
          <label htmlFor="stage" className="label-xs !text-(--muted)">
            Stage
          </label>
          <input
            id="stage"
            type="range"
            min={0}
            max={3}
            step={1}
            value={stage}
            onChange={(e) => setStage(Number(e.target.value))}
            aria-valuetext={STAGE_LABELS[stage]}
            className="mt-4 w-full accent-[var(--snake)]"
          />
          <div className="mt-1 flex justify-between text-[12px] text-(--muted)" aria-hidden="true">
            {STAGE_LABELS.map((s, i) => (
              <button key={s} type="button" tabIndex={-1} onClick={() => setStage(i)} className={`rounded px-1 transition-colors ${i === stage ? 'text-(--fg)' : 'hover:text-(--fg)'}`}>
                {s}
              </button>
            ))}
          </div>
        </div>

        <p className="text-[13px] leading-[1.6] text-(--muted)">
          {model.schedule}.{note ? ` ${note}` : ''}
        </p>
        {fixedArch ? null : (
          <Link href={`/models/${slugOf(model.arch)}`} className="inline-block text-[14px] underline underline-offset-4 decoration-(--line-strong) hover:decoration-current">
            Full page for {model.name}
          </Link>
        )}
      </div>

      <div>
        {layers ? (
          <ModelStrip
            key={model.arch}
            layers={layers}
            cut={cut}
            quant={quant}
            label={`${model.name}: channels per layer at the ${STAGE_LABELS[stage]} stage.`}
            className="h-[clamp(140px,20vw,260px)] w-full"
          />
        ) : (
          <div className="flex h-[clamp(140px,20vw,260px)] items-center justify-center rounded-2xl border border-dashed border-(--line-strong) px-6 text-center text-[14px] text-(--muted)">
            Per-layer channel counts were not recorded for {model.name}, so there is no channel view. The numbers beside it are
            the recorded measurements.
          </div>
        )}

        <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-2xl bg-(--line) sm:grid-cols-4">
          <Gauge label="Accuracy" note={stage ? `${((b.accuracy - row.accuracy) * 100).toFixed(2)} pp vs baseline` : 'baseline'}>
            <MotionNumber value={acc} format={fmtPct} initial={row.accuracy} />
          </Gauge>
          <Gauge label="Size" note={stage ? `${ratio(b.size_mb / row.size_mb)} smaller` : 'baseline'}>
            <MotionNumber value={size} format={fmtMb} initial={row.size_mb} />
          </Gauge>
          <Gauge label="Median latency" note={stage ? `${ratio(b.median_ms / row.median_ms)} faster` : 'baseline'}>
            <MotionNumber value={lat} format={fmtMs} initial={row.median_ms} />
          </Gauge>
          <Gauge label="Parameters" note={stage ? `${ratio(b.parameters / row.parameters)} fewer` : 'baseline'}>
            <MotionNumber value={par} format={fmtCount} initial={row.parameters} />
          </Gauge>
        </dl>

        <p className="mt-4 text-[12px] leading-snug text-(--muted)">
          {STAGE_LABELS[stage]} stage, {row.precision.toUpperCase()}, measured on the exported ONNX file
          {model.host.int8_speed === 'slow' ? ' on a CPU without fast INT8, so this latency is not a deployment figure' : ''}.
          Accuracy cost is always shown next to the ratio it paid for.
        </p>
      </div>
    </div>
  );
}

function Gauge({ label, note, children }: { label: string; note: string; children: React.ReactNode }) {
  return (
    <div className="bg-(--bg) p-4 sm:p-5">
      <dt className="label-xs !text-(--muted)">{label}</dt>
      <dd className="mt-2 text-[clamp(20px,2.3vw,32px)] leading-none tracking-[-0.03em]">{children}</dd>
      <p className="mt-2 text-[12px] leading-snug text-(--muted)">{note}</p>
    </div>
  );
}

export type { ModelRecord };
