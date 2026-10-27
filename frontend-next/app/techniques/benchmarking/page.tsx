import type { Metadata } from 'next';
import { hosts, models, finalStage, ms, pct } from '@/src/data/results';
import { HONESTY_RULES } from '@/src/site/rules';
import { ParetoMini } from '@/src/site/charts/ParetoMini';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'Benchmarking — compresskit' };

export default function BenchmarkingPage() {
  const runs = models.filter((m) => !m.schedule.includes('short smoke run'));
  return (
    <TechniqueShell
      slug="benchmarking"
      title="Every number is measured on the exported file"
      lede={
        <p>
          The most common way to publish a fake speedup is to time two precisions under different settings, or to
          report a FLOP count as if it were a latency. These rules are enforced in the toolkit, not just stated.
        </p>
      }
    >
      <Section
        title="The measurement protocol"
        lede={
          <p>
            Every stage is exported to ONNX and timed through the identical ONNX Runtime session: one intra-op thread,
            batch size 1, full graph optimization, 50 warm-up runs and then 200 timed runs. The median is the headline
            and the 95th percentile shows the tail. Accuracy is top-1 on the held-out test split, evaluated on the
            exported file.
          </p>
        }
      />

      <Section tone="dark" title="Five rules">
        <ol className="grid gap-x-12 gap-y-8 md:grid-cols-2">
          {HONESTY_RULES.map((r, i) => (
            <li key={r.title} className="border-t border-(--line) pt-5">
              <p className="text-[12px] text-(--muted)">Rule {i + 1}</p>
              <h3 className="mt-1 text-[22px] tracking-[-0.03em]">{r.title}</h3>
              <p className="mt-2 text-[15px] leading-relaxed text-(--muted)">{r.body}</p>
            </li>
          ))}
        </ol>
      </Section>

      <Section
        title="Accuracy against latency, per model"
        lede={
          <p>
            Each model has its own plot because latency on different models and different machines does not share an
            axis. The path from the baseline to the final file shows what each stage cost and bought.
          </p>
        }
      >
        <div className="grid gap-6 lg:grid-cols-2">
          {runs.map((m) => (
            <Figure
              key={m.arch}
              caption={`${m.name}: ${pct(m.baseline.accuracy)} at ${ms(m.baseline.median_ms)} to ${pct(finalStage(m).accuracy)} at ${ms(finalStage(m).median_ms)}.`}
            >
              <ParetoMini stages={[m.baseline, ...m.stages]} className="w-full" />
              <Provenance p={m.provenance} source={m.source} />
            </Figure>
          ))}
        </div>
      </Section>

      <Section title="Where it was measured">
        <ul className="grid gap-4 md:grid-cols-2">
          {hosts.map((h) => (
            <li key={h.label} className="rounded-2xl bg-(--chip) p-5">
              <p className="text-[12px] text-(--muted)">{h.label}</p>
              <p className="mt-1 text-[19px] tracking-[-0.02em]">{h.cpu}</p>
              <p className="mt-2 text-[13px] text-(--muted)">
                {h.logical_cores} logical cores, INT8 kernels: {h.int8_speed}
              </p>
            </li>
          ))}
        </ul>
      </Section>
    </TechniqueShell>
  );
}
