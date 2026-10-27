import type { ReactNode } from 'react';

/**
 * A page band. The tone engine (src/landing/tone) reads every [data-tone-band]
 * and cross-fades the whole page between light and dark as bands cross the
 * switch line, so landing and inner pages scroll the same way.
 */
export function Band({
  id,
  tone = 'light',
  className = '',
  children,
  labelledBy,
}: {
  id?: string;
  tone?: 'light' | 'dark';
  className?: string;
  children: ReactNode;
  labelledBy?: string;
}) {
  return (
    <section
      id={id}
      data-tone-band
      data-tone={tone}
      aria-labelledby={labelledBy}
      className={`relative scroll-mt-[72px] ${className}`}
    >
      {children}
    </section>
  );
}

/** Shared section rhythm: the same vertical padding and 1320px column on every page. */
export function Inner({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`mx-auto w-full max-w-[1320px] px-5 py-20 sm:px-8 sm:py-28 ${className}`}>{children}</div>;
}
