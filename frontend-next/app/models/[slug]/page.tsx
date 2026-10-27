import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { Suspense } from 'react';
import { Download } from 'lucide-react';
import { distillation, finalStage, mb, modelBySlug, models, ms, pct, ratio, slugOf, count } from '@/src/data/results';
import { PageFrame } from '@/src/site/PageFrame';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section, Wrap } from '@/src/site/Section';
import { ChannelBars } from '@/src/site/charts/ChannelBars';
import { ParetoMini } from '@/src/site/charts/ParetoMini';
import { PlayedSawtooth } from '@/src/explore/PlayedSawtooth';
import { StageExplorer } from '@/src/explore/StageExplorer';
import { StripReplay } from '@/src/explore/StripReplay';
import { Band } from '@/src/motion/Band';
import { Reveal } from '@/src/landing/Motion';

export function generateStaticParams() {
  return models.map((m) => ({ slug: slugOf(m.arch) }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const m = modelBySlug(slug);
  return { title: m ? `${m.name} compressed — compresskit` : 'Model — compresskit' };
}

async function ModelBody({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const m = modelBySlug(slug);
  if (!m) notFound();
  const f = finalStage(m);
  const ps = m.prune_stats;
  const channelRows = ps ? Object.entries(ps.channels).map(([name, [before, after]]) => ({ name, before, after })) : [];
  const classNames = m.dataset === 'CIFAR-10' ? distillation.classes : null;
  const perClass =
    classNames && m.baseline.recall && f.recall
      ? classNames
          .map((name, i) => ({ name, before: m.baseline.recall![i], after: f.recall![i], support: f.support?.[i] ?? 0 }))
          .sort((a, b) => a.after - a.before - (b.after - b.before))
      : null;

  return (
    <>
      <Band tone="light" labelledBy="model-title">
        <Wrap className="pb-16 pt-24 sm:pb-20 sm:pt-32"><Reveal>
          <p className="label-xs">
            {m.dataset} · {m.provenance.kind}
          </p>
          <h1 id="model-title" className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,4.8vw,66px)] font-normal leading-[1.03] tracking-[-0.045em]">
            {m.name}, compressed
          </h1>
          <p className="mt-6 max-w-[62ch] text-[17px] leading-[1.65] text-(--muted)">{m.schedule}.</p>
        </Reveal></Wrap>
      </Band>

      <Section title="Result">
        <dl className="figures-tab grid gap-8 sm:grid-cols-3">
          <div>
            <dt className="text-[13px] text-(--muted)">Top-1 on the test split</dt>
            <dd className="mt-1 text-[clamp(32px,3.4vw,48px)] leading-none tracking-[-0.04em]">{pct(f.accuracy)}</dd>
            <dd className="mt-2 text-[13px] text-(--muted)">
              from {pct(m.baseline.accuracy)}, a drop of {m.totals.accuracy_drop_pp.toFixed(2)} points
            </dd>
          </div>
          <div>
            <dt className="text-[13px] text-(--muted)">File size</dt>
            <dd className="mt-1 text-[clamp(32px,3.4vw,48px)] leading-none tracking-[-0.04em]">{ratio(m.totals.size_reduction)} smaller</dd>
            <dd className="mt-2 text-[13px] text-(--muted)">
              {mb(m.baseline.size_mb)} to {mb(f.size_mb)}, for that {m.totals.accuracy_drop_pp.toFixed(2)}-point drop
            </dd>
          </div>
          <div>
            <dt className="text-[13px] text-(--muted)">Median latency</dt>
            <dd className="mt-1 text-[clamp(32px,3.4vw,48px)] leading-none tracking-[-0.04em]">{ratio(m.totals.speedup)} faster</dd>
            <dd className="mt-2 text-[13px] text-(--muted)">
              {ms(m.baseline.median_ms)} to {ms(f.median_ms)} on {m.host.cpu}
            </dd>
          </div>
        </dl>
        <Provenance p={m.provenance} source={m.source} />
        {m.provenance.host.includes('Kaggle') ? (
          <p className="mt-3 max-w-[70ch] text-[13px] text-(--muted)">
            This CPU has no fast INT8 kernels, so the latency is not what an INT8-capable CPU would show. Accuracy and
            size are valid on any machine.
          </p>
        ) : null}
      </Section>

      <Section tone="dark" title="Scrub the stages" lede={<p>Move through baseline, pruned, recovered and INT8. Each stop is a measurement of the exported file.</p>}>
        <StageExplorer fixedArch={m.arch} />
      </Section>

      <Section title="Stage by stage">
        <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr] lg:items-center">
          <table className="figures-tab w-full text-left text-[13px]">
            <thead className="text-[11px] uppercase tracking-[0.06em] text-(--muted)">
              <tr>
                <th className="pb-2 font-medium">Stage</th>
                <th className="pb-2 font-medium">Prec.</th>
                <th className="pb-2 text-right font-medium">Top-1</th>
                <th className="pb-2 text-right font-medium">Median</th>
                <th className="pb-2 text-right font-medium">p95</th>
                <th className="pb-2 text-right font-medium">Size</th>
                <th className="pb-2 text-right font-medium">Params</th>
              </tr>
            </thead>
            <tbody>
              {[m.baseline, ...m.stages].map((s) => (
                <tr key={s.stage} className="border-t border-(--line)">
                  <td className="py-2 font-mono text-[12px]">{s.stage}</td>
                  <td className="py-2">{s.precision}</td>
                  <td className="py-2 text-right">{pct(s.accuracy)}</td>
                  <td className="py-2 text-right">{ms(s.median_ms)}</td>
                  <td className="py-2 text-right">{ms(s.p95_ms)}</td>
                  <td className="py-2 text-right">{mb(s.size_mb)}</td>
                  <td className="py-2 text-right">{count(s.parameters)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <ParetoMini stages={[m.baseline, ...m.stages]} className="w-full" />
        </div>
      </Section>

      {m.prune_history.length > 0 ? (
        <Section
          title="Accuracy through the pruning steps"
          lede={<p>Open dot: right after a cut. Solid dot: after fine-tuning that step.</p>}
        >
          <Figure caption={`${m.prune_history.length} steps. Baseline ${pct(m.baseline.accuracy)}.`}>
            <PlayedSawtooth
              baseline={m.baseline.accuracy}
              history={m.prune_history}
              className="w-full max-w-[640px]"
              summary={`Accuracy through ${m.prune_history.length} pruning steps for ${m.name}`}
            />
            <Provenance p={m.provenance} source={m.source} />
          </Figure>
        </Section>
      ) : null}

      {ps && channelRows.length > 0 ? (
        <Section title="Channels per layer" lede={<p>{count(ps.params_before)} parameters became {count(ps.params_after)}; multiply-accumulates went from {(ps.macs_before / 1e6).toFixed(1)}M to {(ps.macs_after / 1e6).toFixed(1)}M.</p>}>
          <div className="mb-6">
            <Figure caption="Each column is a layer and each cell a channel; the recorded cut plays when this scrolls into view.">
              <StripReplay arch={m.arch} />
            </Figure>
          </div>
          <Figure caption="Pale bar: original width. Solid: kept. An asterisk marks a kept count that is not a multiple of 16.">
            <details>
              <summary className="cursor-pointer text-[14px]">Show {channelRows.length} layers</summary>
              <div className="mt-5">
                <ChannelBars rows={channelRows} align={16} />
              </div>
            </details>
          </Figure>
        </Section>
      ) : null}

      {perClass ? (
        <Section
          title="Which classes lost accuracy"
          lede={<p>Top-1 hides where a compressed model gets worse. Recall per class, worst change first.</p>}
        >
          <Figure caption="Recall on the CIFAR-10 test split, original FP32 against the final file.">
            <table className="figures-tab w-full text-left text-[13px]">
              <thead className="text-[11px] uppercase tracking-[0.06em] text-(--muted)">
                <tr>
                  <th className="pb-2 font-medium">Class</th>
                  <th className="pb-2 text-right font-medium">Original</th>
                  <th className="pb-2 text-right font-medium">Compressed</th>
                  <th className="pb-2 text-right font-medium">Change (points)</th>
                </tr>
              </thead>
              <tbody>
                {perClass.map((c) => (
                  <tr key={c.name} className="border-t border-(--line)">
                    <td className="py-2">{c.name}</td>
                    <td className="py-2 text-right">{pct(c.before, 1)}</td>
                    <td className="py-2 text-right">{pct(c.after, 1)}</td>
                    <td className="py-2 text-right">{((c.after - c.before) * 100).toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Provenance p={m.provenance} source={m.source} />
          </Figure>
        </Section>
      ) : null}

      {m.downloads && m.downloads.length > 0 ? (
        <Section title="Download" lede={<p>The actual files from this run. Open them in any ONNX viewer or runtime.</p>}>
          <ul className="flex flex-wrap gap-3">
            {m.downloads.map((d) => (
              <li key={d.href}>
                <a
                  href={d.href}
                  download
                  className="inline-flex h-12 items-center gap-3 rounded-full border border-(--line-strong) px-5 text-[14px] transition-colors hover:bg-(--hover)"
                >
                  <Download className="h-4 w-4" aria-hidden="true" />
                  {d.label}
                  <span className="text-(--muted)">{(d.size_bytes / 1e6).toFixed(2)} MB</span>
                </a>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {m.warnings.length > 0 ? (
        <Section title="Notes from the run">
          <ul className="space-y-2 text-[14px] text-(--muted)">
            {m.warnings.map((w) => (
              <li key={w} className="border-l-2 border-(--line-strong) pl-4">
                {w}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </>
  );
}

export default function ModelPage({ params }: { params: Promise<{ slug: string }> }) {
  return (
    <PageFrame>
      <Suspense fallback={<div className="min-h-[70svh]" aria-busy="true" />}>
        <ModelBody params={params} />
      </Suspense>
    </PageFrame>
  );
}
