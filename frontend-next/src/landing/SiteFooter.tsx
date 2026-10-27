import Link from 'next/link';
import { Fragment } from 'react';
import { Scissors } from 'lucide-react';
import { Mark } from './Mark';
import { TECHNIQUES } from '@/src/site/techniques';

function MarqueeRun({ hidden }: { hidden?: boolean }) {
  return (
    <span aria-hidden={hidden ? true : undefined} className="flex shrink-0 items-center">
      {[0, 1, 2].map((n) => (
        <Fragment key={n}>
          <span className="bg-gradient-to-b from-brand-200 via-brand-400 to-brand-600 bg-clip-text px-[0.12em] text-transparent">
            Prune, then measure
          </span>
          <span className="mx-[0.12em] flex h-[0.62em] w-[0.62em] shrink-0 items-center justify-center rounded-full bg-[radial-gradient(circle_at_32%_28%,var(--color-brand-100),var(--color-brand-400)_45%,var(--color-brand-600))] transition-transform duration-500 group-hover:rotate-[-20deg]">
            <Scissors className="h-[0.28em] w-[0.28em] text-ink-950" strokeWidth={2.2} aria-hidden="true" />
          </span>
        </Fragment>
      ))}
    </span>
  );
}

/** `compact` drops the marquee and the overlap with the section above, for inner pages. */
export function SiteFooter({ compact = false }: { compact?: boolean } = {}) {
  return (
    <footer
      id="site-footer"
      data-tone-band
      /* Counts as a light band for the page mix: the footer is a dark panel that owns its colours
         (data-tone-own), so the page never flips from white to black while content is on screen. */
      data-tone="light"
      data-tone-own="dark"
      className={`relative z-10 overflow-hidden bg-(--bg) text-(--fg) ${compact ? '' : '-mt-10 rounded-t-[40px] shadow-[0_-30px_70px_-30px_rgb(20_20_20/0.35)]'}`}
    >
      {compact ? null : (
      <Link
        href="/run"
        className="group block overflow-hidden border-b border-(--line) py-8 sm:py-12"
        aria-label="Run compresskit"
      >
        <span
          className="ss-marquee flex w-max text-[clamp(76px,13vw,212px)] font-medium uppercase leading-[1.05] tracking-[-0.05em]"
          style={{ '--ss-marquee-duration': '30s' } as React.CSSProperties}
        >
          <MarqueeRun />
          <MarqueeRun hidden />
        </span>
      </Link>
      )}

      <div className="mx-auto grid max-w-[1320px] gap-12 px-5 pb-12 pt-16 sm:px-8 md:grid-cols-[1.3fr_1fr_1fr_1.6fr]">
        <div>
          <Link href="/" className="inline-flex items-center gap-2.5 rounded-md">
            <Mark className="h-7 w-7" />
            <span className="text-[16px] font-semibold">compresskit</span>
          </Link>
          <p className="mt-4 max-w-[260px] text-[14px] leading-relaxed text-(--muted)">
            Pruning, distillation and quantization for image-classification CNNs, measured on the exported ONNX file.
          </p>
        </div>

        <nav aria-label="Techniques" className="text-[14px]">
          <p className="text-(--muted)">Techniques</p>
          <ul className="mt-4 space-y-3">
            {TECHNIQUES.map((t) => (
              <li key={t.slug}>
                <Link href={`/techniques/${t.slug}`} className="hover:text-brand-400">
                  {t.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <nav aria-label="Explore" className="text-[14px]">
          <p className="text-(--muted)">Explore</p>
          <ul className="mt-4 space-y-3">
            {[
              ['/pipeline', 'The pipeline'],
              ['/models', 'Compressed models'],
              ['/notebooks', 'Notebooks'],
              ['/run', 'Run it'],
            ].map(([href, label]) => (
              <li key={href}>
                <Link href={href} className="hover:text-brand-400">
                  {label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <div className="space-y-4 text-[14px] leading-relaxed">
          <p>
            <span className="font-semibold">Where the numbers come from.</span>{' '}
            <span className="text-(--muted)">
              Each page names its source run, the machine it ran on, and whether the figures were measured by the
              toolkit or transcribed from a log.
            </span>
          </p>
          <p className="text-(--muted)">
            Latency is a statement about the CPU that measured it. Check it on your target with compresskit bench.
          </p>
        </div>
      </div>

      <div className="mx-auto flex max-w-[1320px] flex-col gap-2 border-t border-(--line) px-5 py-6 text-[13px] text-(--muted) sm:flex-row sm:justify-between sm:px-8">
        <p>compresskit</p>
        <p>MIT licensed per pyproject.toml</p>
      </div>
    </footer>
  );
}
