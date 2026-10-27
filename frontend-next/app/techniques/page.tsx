import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { PageFrame } from '@/src/site/PageFrame';
import { TECHNIQUES } from '@/src/site/techniques';
import { Band, Inner } from '@/src/motion/Band';

export const metadata = { title: 'Techniques | compresskit' };

export default function TechniquesIndex() {
  return (
    <PageFrame>
      <Band tone="light">
        <Inner>
          <p className="label-xs">Techniques</p>
          <h1 className="mt-4 max-w-[16ch] text-balance text-[clamp(40px,5.6vw,84px)] font-normal leading-[1] tracking-[-0.05em]">
            Eight steps, each with its own page.
          </h1>
          <ol className="mt-14 grid gap-px overflow-hidden rounded-[24px] bg-(--line) sm:grid-cols-2">
            {TECHNIQUES.map((t, i) => (
              <li key={t.slug} className="bg-(--bg)">
                <Link href={`/techniques/${t.slug}`} className="group flex h-full items-center justify-between gap-4 p-6 transition-colors hover:bg-(--hover) sm:p-8">
                  <span>
                    <span className="figures-tab text-[12px] text-(--muted)">{String(i + 1).padStart(2, '0')}</span>
                    <span className="mt-1 block text-[22px] tracking-[-0.03em]">{t.label}</span>
                  </span>
                  <ArrowUpRight className="h-5 w-5 text-(--muted) transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden="true" />
                </Link>
              </li>
            ))}
          </ol>
        </Inner>
      </Band>
    </PageFrame>
  );
}
