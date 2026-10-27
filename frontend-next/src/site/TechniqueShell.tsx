import Link from 'next/link';
import type { ReactNode } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { PageFrame } from './PageFrame';
import { Wrap } from './Section';
import { Band } from '@/src/motion/Band';
import { Reveal } from '@/src/landing/Motion';
import { TECHNIQUES, type TechniqueSlug } from './techniques';

/** Header, reading-order navigation and frame shared by every technique page. */
export function TechniqueShell({
  slug,
  title,
  lede,
  children,
}: {
  slug: TechniqueSlug;
  title: string;
  lede: ReactNode;
  children: ReactNode;
}) {
  const index = TECHNIQUES.findIndex((t) => t.slug === slug);
  const prev = TECHNIQUES[index - 1];
  const next = TECHNIQUES[index + 1];
  return (
    <PageFrame>
      <Band tone="light" labelledBy="technique-title">
        <Wrap className="pb-16 pt-24 sm:pb-20 sm:pt-32">
          <Reveal>
            <p className="label-xs">
              Technique {String(index + 1).padStart(2, '0')} of {String(TECHNIQUES.length).padStart(2, '0')}
            </p>
            <h1
              id="technique-title"
              className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,5vw,72px)] font-normal leading-[1.02] tracking-[-0.05em]"
            >
              {title}
            </h1>
            <div className="mt-6 max-w-[60ch] text-[17px] leading-[1.65] text-(--muted)">{lede}</div>
          </Reveal>
          <ol aria-label="All techniques" className="mt-10 flex flex-wrap gap-2">
            {TECHNIQUES.map((t, i) => (
              <li key={t.slug}>
                <Link
                  href={`/techniques/${t.slug}`}
                  aria-current={t.slug === slug ? 'page' : undefined}
                  className="inline-flex h-9 items-center gap-2 rounded-full border border-(--line-strong) px-3.5 text-[13px] transition-colors hover:bg-(--hover) aria-[current]:border-transparent aria-[current]:bg-(--inv-bg) aria-[current]:text-(--inv-fg)"
                >
                  <span className="figures-tab text-[11px] opacity-60">{String(i + 1).padStart(2, '0')}</span>
                  {t.label}
                </Link>
              </li>
            ))}
          </ol>
        </Wrap>
      </Band>

      {children}

      <Band tone="light"><nav aria-label="Technique pages" className="border-t border-(--line)">
        <Wrap className="grid gap-px py-8 sm:grid-cols-2">
          {prev ? (
            <Link href={`/techniques/${prev.slug}`} className="group flex items-center gap-3 rounded-2xl p-4 transition-colors hover:bg-(--hover)">
              <ArrowLeft className="h-4 w-4 shrink-0 transition-transform group-hover:-translate-x-0.5" aria-hidden="true" />
              <span>
                <span className="block text-[12px] text-(--muted)">Previous</span>
                <span className="text-[17px] tracking-[-0.02em]">{prev.label}</span>
              </span>
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link href={`/techniques/${next.slug}`} className="group flex items-center justify-end gap-3 rounded-2xl p-4 text-right transition-colors hover:bg-(--hover)">
              <span>
                <span className="block text-[12px] text-(--muted)">Next</span>
                <span className="text-[17px] tracking-[-0.02em]">{next.label}</span>
              </span>
              <ArrowRight className="h-4 w-4 shrink-0 transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
            </Link>
          ) : (
            <Link href="/pipeline" className="group flex items-center justify-end gap-3 rounded-2xl p-4 text-right transition-colors hover:bg-(--hover) sm:col-start-2">
              <span>
                <span className="block text-[12px] text-(--muted)">Next</span>
                <span className="text-[17px] tracking-[-0.02em]">The whole pipeline</span>
              </span>
              <ArrowRight className="h-4 w-4 shrink-0" aria-hidden="true" />
            </Link>
          )}
        </Wrap>
      </nav></Band>
    </PageFrame>
  );
}
