import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { Suspense } from 'react';
import { notebooks } from '@/src/data/results';
import { Code } from '@/src/site/Code';
import { MiniMarkdown } from '@/src/site/MiniMarkdown';
import { Band } from '@/src/motion/Band';
import { PageFrame } from '@/src/site/PageFrame';
import { Section, Wrap } from '@/src/site/Section';

const REPO_BLOB = 'https://github.com/Bhagyesh0603/Model-Compression-ToolKit./blob/master';

export function generateStaticParams() {
  return notebooks.map((n) => ({ slug: n.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const n = notebooks.find((x) => x.slug === slug);
  return { title: n ? `${n.title} notebook — compresskit` : 'Notebook — compresskit' };
}

async function NotebookBody({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const n = notebooks.find((x) => x.slug === slug);
  if (!n) notFound();
  const codeCells = n.cells.filter((c) => c.type === 'code').length;

  return (
    <>
      <Band tone="light">
        <Wrap className="pb-16 pt-24 sm:pb-20 sm:pt-32">
          <p className="label-xs">Notebook · {n.cells.length} cells, {codeCells} of them code</p>
          <h1 className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,4.8vw,66px)] font-normal leading-[1.03] tracking-[-0.05em]">
            {n.title}
          </h1>
          <p className="mt-6 max-w-[62ch] text-[17px] leading-[1.65] text-(--muted)">{n.blurb}</p>
          <p className="mt-6 flex flex-wrap gap-3 text-[14px]">
            <a
              href={`${REPO_BLOB}/${n.path}`}
              className="inline-flex h-11 items-center rounded-full bg-(--inv-bg) px-5 font-medium text-(--inv-fg) transition-colors hover:bg-(--inv-hover)"
            >
              View the notebook file
            </a>
            <span className="inline-flex h-11 items-center font-mono text-[12px] text-(--muted)">{n.path}</span>
          </p>
        </Wrap>
      </Band>

      <Section
        title="To run it"
        lede={
          <ol className="list-decimal space-y-1.5 pl-5">
            <li>On Kaggle choose Create, New Notebook, then File, Import Notebook, and upload the file.</li>
            <li>In the right panel set the accelerator to a GPU and Internet to On.</li>
            <li>Edit the config cell if you want different settings, then Run All.</li>
            <li>Download the files from the Output tab.</li>
          </ol>
        }
      />

      <Section title="Every cell, in order">
        <ol className="space-y-5">
          {n.cells.map((c, i) => (
            <li key={i} className="grid gap-3 lg:grid-cols-[8rem_1fr]">
              <p className="font-mono text-[12px] text-(--muted)">
                {String(i + 1).padStart(2, '0')} · {c.type}
              </p>
              {c.type === 'markdown' ? <MiniMarkdown source={c.source} /> : <Code>{c.source}</Code>}
            </li>
          ))}
        </ol>
      </Section>
    </>
  );
}

export default function NotebookPage({ params }: { params: Promise<{ slug: string }> }) {
  return (
    <PageFrame>
      <Suspense fallback={<div className="min-h-[70svh]" aria-busy="true" />}>
        <NotebookBody params={params} />
      </Suspense>
    </PageFrame>
  );
}
