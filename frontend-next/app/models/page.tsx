import type { Metadata } from 'next';
import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { finalStage, mb, models, pct, ratio, slugOf } from '@/src/data/results';
import { PageFrame } from '@/src/site/PageFrame';
import { Wrap } from '@/src/site/Section';

export const metadata: Metadata = { title: 'Compressed models — compresskit' };

export default function ModelsPage() {
  return (
    <PageFrame>
      <Wrap className="pb-10 pt-16 sm:pt-24">
        <p className="label-xs">Compressed models</p>
        <h1 className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,4.8vw,66px)] font-normal leading-[1.03] tracking-[-0.045em]">
          What the pipeline produced
        </h1>
        <p className="mt-6 max-w-[60ch] text-[17px] leading-[1.65] text-(--muted)">
          One page per recorded run, each with the stage table, the accuracy cost of every size and speed gain, and where
          the numbers came from.
        </p>
      </Wrap>
      <Wrap className="pb-24">
        <ul className="grid gap-4 md:grid-cols-2">
          {models.map((m) => {
            const f = finalStage(m);
            const short = m.schedule.includes('short smoke run');
            return (
              <li key={m.arch}>
                <Link href={`/models/${slugOf(m.arch)}`} className="group block h-full rounded-[22px] border border-(--line) p-6 transition-colors hover:bg-(--hover)">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-[12px] text-(--muted)">{m.dataset}</p>
                      <h2 className="mt-1 text-[28px] tracking-[-0.03em]">{m.name}</h2>
                    </div>
                    <span className="flex h-9 w-9 items-center justify-center rounded-full bg-(--chip) transition-all duration-300 group-hover:rotate-45 group-hover:bg-brand-600 group-hover:text-white">
                      <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                    </span>
                  </div>
                  <dl className="figures-tab mt-6 grid grid-cols-3 gap-4">
                    <div>
                      <dt className="text-[12px] text-(--muted)">Size</dt>
                      <dd className="mt-1 text-[22px] tracking-[-0.03em]">{ratio(m.totals.size_reduction)}</dd>
                      <dd className="text-[11px] text-(--muted)">{mb(m.baseline.size_mb)} to {mb(f.size_mb)}</dd>
                    </div>
                    <div>
                      <dt className="text-[12px] text-(--muted)">Top-1</dt>
                      <dd className="mt-1 text-[22px] tracking-[-0.03em]">{pct(f.accuracy)}</dd>
                      <dd className="text-[11px] text-(--muted)">from {pct(m.baseline.accuracy)}</dd>
                    </div>
                    <div>
                      <dt className="text-[12px] text-(--muted)">Latency</dt>
                      <dd className="mt-1 text-[22px] tracking-[-0.03em]">{ratio(m.totals.speedup)}</dd>
                      <dd className="text-[11px] text-(--muted)">{m.provenance.host.split(' (')[0]}</dd>
                    </div>
                  </dl>
                  <p className="mt-5 text-[12px] leading-snug text-(--muted)">
                    {short ? 'Short test run. ' : ''}
                    {m.schedule}. {m.provenance.kind === 'transcribed' ? 'Typed in from the run log.' : 'Measured by the toolkit.'}
                  </p>
                </Link>
              </li>
            );
          })}
        </ul>
      </Wrap>
    </PageFrame>
  );
}
