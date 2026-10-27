import type { Metadata } from 'next';
import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { notebooks } from '@/src/data/results';
import { PageFrame } from '@/src/site/PageFrame';
import { Wrap } from '@/src/site/Section';

export const metadata: Metadata = { title: 'Notebooks — compresskit' };

export default function NotebooksPage() {
  return (
    <PageFrame>
      <Wrap className="pb-10 pt-16 sm:pt-24">
        <p className="label-xs">Notebooks</p>
        <h1 className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,4.8vw,66px)] font-normal leading-[1.03] tracking-[-0.045em]">
          Run all, on a free GPU
        </h1>
        <p className="mt-6 max-w-[60ch] text-[17px] leading-[1.65] text-(--muted)">
          Two Kaggle notebooks run the whole flow top to bottom. The cell listings below are read from the notebook files
          in the repository, so they match what you will run.
        </p>
      </Wrap>
      <Wrap className="pb-24">
        <ul className="grid gap-4 md:grid-cols-2">
          {notebooks.map((n) => (
            <li key={n.slug}>
              <Link href={`/notebooks/${n.slug}`} className="group block h-full rounded-[22px] border border-(--line) p-6 transition-colors hover:bg-(--hover)">
                <div className="flex items-start justify-between">
                  <h2 className="text-[28px] tracking-[-0.03em]">{n.title}</h2>
                  <span className="flex h-9 w-9 items-center justify-center rounded-full bg-(--chip) transition-all duration-300 group-hover:rotate-45 group-hover:bg-brand-600 group-hover:text-white">
                    <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                  </span>
                </div>
                <p className="mt-3 text-[15px] leading-relaxed text-(--muted)">{n.blurb}</p>
                <p className="mt-5 font-mono text-[12px] text-(--muted)">{n.path} · {n.cells.length} cells</p>
              </Link>
            </li>
          ))}
        </ul>
      </Wrap>
    </PageFrame>
  );
}
