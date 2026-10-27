'use client';

import { useRef, useState, type ReactNode } from 'react';
import { AnimatePresence, animate, motion, useMotionValue, useMotionValueEvent, useScroll, type MotionValue } from 'motion/react';
import { useLenis } from 'lenis/react';
import { useMediaQuery, usePrefersReducedMotion } from '@/src/landing/hooks';
import { DUR, EASE } from './tokens';

export type ScrollyStep = { id: string; kicker: string; title: string; body: ReactNode };

export type ScrollyState = {
  /** 0..1 over the whole pin. Scroll-linked on desktop, tweened to the chosen step otherwise. */
  progress: MotionValue<number>;
  step: number;
  steps: number;
};

const PER_STEP_SVH = 85;

/**
 * The one scroll primitive. On md+ it pins a visual beside the step text and
 * maps scroll to a progress number; below md, or with reduced motion, the same
 * visual is shown with a step selector instead, so nothing depends on scrolling.
 */
export function Scrolly({
  steps,
  visual,
  label,
  header,
}: {
  header?: ReactNode;
  steps: ScrollyStep[];
  visual: (state: ScrollyState) => ReactNode;
  label: string;
}) {
  const reduce = usePrefersReducedMotion();
  const wide = useMediaQuery('(min-width: 768px)', true);
  const pinned = wide && !reduce;
  const track = useRef<HTMLDivElement>(null);
  const lenis = useLenis();
  const n = steps.length;

  const { scrollYProgress } = useScroll({ target: track, offset: ['start start', 'end end'] });
  const manual = useMotionValue(1 / (2 * n));
  const progress = pinned ? scrollYProgress : manual;
  const [step, setStep] = useState(0);

  useMotionValueEvent(progress, 'change', (p) => {
    const next = Math.min(n - 1, Math.max(0, Math.floor(p * n)));
    setStep((cur) => (cur === next ? cur : next));
  });

  const goTo = (i: number) => {
    if (!pinned) {
      setStep(i);
      animate(manual, (i + 0.5) / n, { duration: DUR.layout, ease: EASE });
      return;
    }
    const el = track.current;
    if (!el) return;
    const span = el.offsetHeight - window.innerHeight;
    const y = el.getBoundingClientRect().top + window.scrollY + ((i + 0.5) / n) * span;
    if (lenis) lenis.scrollTo(y, { duration: 1.1 });
    else window.scrollTo({ top: y, behavior: 'smooth' });
  };

  const active = steps[step];
  const text = (
    <div className="relative min-h-[260px]">
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={active.id}
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: DUR.enter, ease: EASE }}
        >
          <p className="label-xs !text-(--muted)">{active.kicker}</p>
          <h3 className="mt-3 text-[clamp(28px,3.2vw,46px)] font-normal leading-[1.05] tracking-[-0.04em]">{active.title}</h3>
          <div className="mt-5 max-w-[46ch] text-[16px] leading-[1.7] text-(--muted)">{active.body}</div>
        </motion.div>
      </AnimatePresence>
    </div>
  );

  const rail = (
    <div role="tablist" aria-label={label} className="mt-8 flex flex-wrap gap-2">
      {steps.map((s, i) => (
        <button
          key={s.id}
          role="tab"
          type="button"
          aria-selected={i === step}
          onClick={() => goTo(i)}
          className="group flex h-9 items-center gap-2 rounded-full border border-(--line-strong) px-3.5 text-[13px] transition-colors aria-selected:border-transparent aria-selected:bg-(--inv-bg) aria-selected:text-(--inv-fg) hover:bg-(--hover) aria-selected:hover:bg-(--inv-bg)"
        >
          <span className="figures-tab text-[11px] opacity-60">{String(i + 1).padStart(2, '0')}</span>
          {s.kicker}
        </button>
      ))}
    </div>
  );

  const state: ScrollyState = { progress, step, steps: n };

  if (!pinned) {
    return (
      <div className="mx-auto w-full max-w-[1320px] px-5 py-16 sm:px-8">
        {header ? <div className="mb-10">{header}</div> : null}
        <div className="mb-8">{visual(state)}</div>
        {text}
        {rail}
      </div>
    );
  }

  return (
    <div ref={track} style={{ height: `${n * PER_STEP_SVH}svh` }} className="relative">
      <div className="sticky top-[72px] flex h-[calc(100svh-72px)] flex-col justify-center gap-10 py-8">
        {header ? <div className="mx-auto w-full max-w-[1320px] px-8">{header}</div> : null}
        <div className="mx-auto grid w-full max-w-[1320px] items-center gap-14 px-8 lg:grid-cols-[minmax(0,4.4fr)_minmax(0,7.6fr)]">
          <div>
            {text}
            {rail}
          </div>
          <div>{visual(state)}</div>
        </div>
      </div>
    </div>
  );
}
