import type { Metadata } from 'next';
import Link from 'next/link';
import { finalStage, models, ms, mb, pct, ratio } from '@/src/data/results';
import { Band } from '@/src/motion/Band';
import { PageFrame } from '@/src/site/PageFrame';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section, Wrap } from '@/src/site/Section';
import { ParetoMini } from '@/src/site/charts/ParetoMini';

export const metadata: Metadata = { title: 'The pipeline — compresskit' };

const STAGES = [
  ['Prune', 'Remove channels through the dependency graph, widths aligned for SIMD.', '/techniques/pruning'],
  ['Recover', 'Re-estimate BatchNorm statistics, then fine-tune against the original as teacher.', '/techniques/recalibration'],
  ['Distill', 'Optional extra distillation epochs once the architecture is final.', '/techniques/distillation'],
  ['Quantize', 'INT8 where the CPU has fast INT8, FP16 where it does not. Always last.', '/techniques/quantization'],
  ['Verify', 'Export to ONNX and measure accuracy and latency on that file.', '/techniques/benchmarking'],
] as const;

export default function PipelinePage() {
  const runs = models.filter((m) => !m.schedule.includes('short smoke run'));
  return (
    <PageFrame>
      <Band tone="light">
        <Wrap className="pb-16 pt-24 sm:pb-20 sm:pt-32">
          <p className="label-xs">Pipeline</p>
          <h1 className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,4.8vw,66px)] font-normal leading-[1.03] tracking-[-0.05em]">
            Five stages in a fixed order
          </h1>
          <p className="mt-6 max-w-[60ch] text-[17px] leading-[1.65] text-(--muted)">
            Every stage exports an ONNX file and is measured, so the result is a table of real numbers per stage rather
            than one number out of a black box. If latency got worse, you can see which stage did it.
          </p>
        </Wrap>
      </Band>

      <Section title="The stages">
        <ol className="grid gap-px overflow-hidden rounded-[22px] border border-(--line) bg-(--line) md:grid-cols-5">
          {STAGES.map(([name, body, href], i) => (
            <li key={name} className="bg-(--bg)">
              <Link href={href} className="block h-full p-5 transition-colors hover:bg-(--hover)">
                <p className="text-[12px] text-(--muted)">Stage {i + 1}</p>
                <p className="mt-1 text-[19px] tracking-[-0.02em]">{name}</p>
                <p className="mt-2 text-[13px] leading-snug text-(--muted)">{body}</p>
              </Link>
            </li>
          ))}
        </ol>
      </Section>

      <Section
        title="Why quantization is last"
        lede={
          <p>
            Quantization changes the weight type from float to integer. After that, structural pruning no longer
            applies and gradient-based fine-tuning is blocked. The pipeline refuses a plan where quantize is not the
            final stage, instead of reordering it silently.
          </p>
        }
      />

      <Section
        title="Two recorded runs, stage by stage"
        lede={
          <p>
            The same pipeline on two different networks and two different machines. The ratio always appears next to
            the accuracy it cost.
          </p>
        }
      >
        <div className="space-y-6">
          {runs.map((m) => {
            const f = finalStage(m);
            return (
              <Figure
                key={m.arch}
                caption={`${m.name} on ${m.dataset}: ${ratio(m.totals.size_reduction)} smaller (${mb(m.baseline.size_mb)} to ${mb(f.size_mb)}) at a cost of ${m.totals.accuracy_drop_pp.toFixed(2)} points of top-1; ${ratio(m.totals.speedup)} faster on ${m.host.cpu}. Schedule: ${m.schedule}.`}
              >
                <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr] lg:items-center">
                  <table className="figures-tab w-full text-left text-[13px]">
                    <thead className="text-[11px] uppercase tracking-[0.06em] text-(--muted)">
                      <tr>
                        <th className="pb-2 font-medium">Stage</th>
                        <th className="pb-2 font-medium">Prec.</th>
                        <th className="pb-2 text-right font-medium">Top-1</th>
                        <th className="pb-2 text-right font-medium">Latency</th>
                        <th className="pb-2 text-right font-medium">Size</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[m.baseline, ...m.stages].map((s) => (
                        <tr key={s.stage} className="border-t border-(--line)">
                          <td className="py-2 font-mono text-[12px]">{s.stage}</td>
                          <td className="py-2">{s.precision}</td>
                          <td className="py-2 text-right">{pct(s.accuracy)}</td>
                          <td className="py-2 text-right">{ms(s.median_ms)}</td>
                          <td className="py-2 text-right">{mb(s.size_mb)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <ParetoMini stages={[m.baseline, ...m.stages]} className="w-full" />
                </div>
                <Provenance p={m.provenance} source={m.source} />
              </Figure>
            );
          })}
        </div>
      </Section>
    </PageFrame>
  );
}
