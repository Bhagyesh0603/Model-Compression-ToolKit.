import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { finalStage, mb, models, ms, pct, ratio, slugOf, type ModelRecord } from '@/src/data/results';
import { Band, Inner } from '@/src/motion/Band';
import { Reveal } from './Motion';
import { AnimBar } from '@/src/viz/AnimBar';

/** Three recorded runs side by side. Each ratio sits beside the accuracy it cost. */
export function ModelsCompare() {
  return (
    <Band id="models" tone="light" labelledBy="models-title">
      <Inner>
        <Reveal>
          <p className="label-xs">The results</p>
          <h2 id="models-title" className="mt-4 max-w-[20ch] text-balance text-[clamp(32px,4.2vw,60px)] font-normal leading-[1.04] tracking-[-0.045em]">
            Three models, and what each one paid.
          </h2>
        </Reveal>
        <div className="mt-12 grid gap-5 lg:grid-cols-3">
          {models.map((m, i) => (
            <Reveal key={m.arch} delay={i * 0.08}>
              <Card model={m} />
            </Reveal>
          ))}
        </div>
      </Inner>
    </Band>
  );
}

function Card({ model }: { model: ModelRecord }) {
  const b = model.baseline;
  const f = finalStage(model);
  return (
    <Link
      href={`/models/${slugOf(model.arch)}`}
      className="group flex h-full flex-col rounded-[24px] border border-(--line) p-6 transition-colors hover:bg-(--hover)"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-[24px] tracking-[-0.03em]">{model.name}</h3>
          <p className="mt-1 text-[13px] text-(--muted)">{model.dataset}</p>
        </div>
        <ArrowUpRight className="h-5 w-5 text-(--muted) transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden="true" />
      </div>
      <p className="mt-4 inline-flex w-fit items-center gap-2 rounded-full bg-(--chip) px-3 py-1 text-[12px] text-(--chip-fg)">
        <span className={`h-1.5 w-1.5 rounded-full ${model.provenance.kind === 'measured' ? 'bg-(--snake)' : 'bg-int8'}`} aria-hidden="true" />
        {model.provenance.kind} · {model.provenance.host}
      </p>
      <div className="mt-6 space-y-5">
        <div className="space-y-2">
          <AnimBar label="Size, before" value={1} display={mb(b.size_mb)} />
          <AnimBar label={`Size, after (${ratio(model.totals.size_reduction)} smaller)`} value={f.size_mb / b.size_mb} display={mb(f.size_mb)} accent />
        </div>
        <div className="space-y-2">
          <AnimBar label="Latency, before" value={1} display={ms(b.median_ms)} />
          <AnimBar label={`Latency, after (${ratio(model.totals.speedup)} faster)`} value={f.median_ms / b.median_ms} display={ms(f.median_ms)} accent />
        </div>
        <div className="space-y-2">
          <AnimBar label="Accuracy, before" value={b.accuracy} display={pct(b.accuracy)} />
          <AnimBar label={`Accuracy, after (${model.totals.accuracy_drop_pp.toFixed(2)} pp lower)`} value={f.accuracy} display={pct(f.accuracy)} accent />
        </div>
      </div>
      <p className="mt-auto pt-6 text-[12px] leading-snug text-(--muted)">
        {model.schedule}.
        {model.host.int8_speed === 'slow' ? ' Latency is from a CPU without fast INT8, so it is not a deployment figure.' : ''}
      </p>
    </Link>
  );
}
