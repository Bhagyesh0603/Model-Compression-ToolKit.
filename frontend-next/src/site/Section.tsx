import type { ReactNode } from 'react';
import { Reveal } from '@/src/landing/Motion';
import { Band } from '@/src/motion/Band';

const WRAP = 'mx-auto w-full max-w-[1320px] px-5 sm:px-8';

export function Wrap({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`${WRAP} ${className}`}>{children}</div>;
}

/**
 * A titled band on an inner page. It is a tone band like the landing sections,
 * so a dark data stage and the light text around it cross-fade the same way.
 * The text column stays readable; figures use the full width.
 */
export function Section({
  title,
  lede,
  children,
  id,
  tone = 'light',
}: {
  title: string;
  lede?: ReactNode;
  children?: ReactNode;
  id?: string;
  tone?: 'light' | 'dark';
}) {
  return (
    <Band id={id} tone={tone}>
      <div className={`${WRAP} py-20 sm:py-28`}>
        <Reveal>
          <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr] lg:gap-14">
            <h2 className="text-[clamp(26px,2.8vw,40px)] font-normal leading-[1.08] tracking-[-0.04em]">{title}</h2>
            {lede ? <div className="max-w-[62ch] text-[16px] leading-[1.7] text-(--muted)">{lede}</div> : null}
          </div>
        </Reveal>
        {children ? (
          <Reveal delay={0.08} className="mt-12">
            {children}
          </Reveal>
        ) : null}
      </div>
    </Band>
  );
}

/** A chart or table with its caption. */
export function Figure({ caption, children }: { caption: ReactNode; children: ReactNode }) {
  return (
    <figure className="rounded-[24px] border border-(--line) p-5 sm:p-8">
      <div className="text-(--fg)">{children}</div>
      <figcaption className="mt-5 text-[13px] leading-snug text-(--muted)">{caption}</figcaption>
    </figure>
  );
}
