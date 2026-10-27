'use client';

import { useEffect, useRef } from 'react';
import { RotateCcw } from 'lucide-react';
import { animate, useInView, useMotionValue } from 'motion/react';
import type { PruneStep } from '@/src/data/results';
import { usePrefersReducedMotion } from '@/src/landing/hooks';
import { EASE } from '@/src/motion/tokens';
import { AnimSawtooth } from '@/src/viz/AnimSawtooth';

/** The recorded accuracy-through-cuts line, drawn once when it scrolls into view and replayable. */
export function PlayedSawtooth({
  baseline,
  history,
  summary,
  className,
}: {
  baseline: number;
  history: Pick<PruneStep, 'step' | 'acc_after_cut' | 'acc_after_finetune'>[];
  summary: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const seen = useInView(ref, { once: true, margin: '0px 0px -15% 0px' });
  const reduce = usePrefersReducedMotion();
  const reveal = useMotionValue(reduce ? 1 : 0);
  const play = () => {
    reveal.set(0);
    return animate(reveal, 1, { duration: 0.7 * history.length + 0.8, ease: EASE });
  };
  useEffect(() => {
    if (!seen) return;
    if (reduce) {
      reveal.set(1);
      return;
    }
    const c = play();
    return () => c.stop();
  }, [seen, reduce]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div ref={ref}>
      <AnimSawtooth baseline={baseline} history={history} reveal={reveal} summary={summary} className={className} />
      <button
        type="button"
        onClick={() => play()}
        className="mt-3 inline-flex h-8 items-center gap-1.5 rounded-full bg-(--chip) px-3 text-[12px] text-(--chip-fg) transition-colors hover:bg-(--chip-hover)"
      >
        <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
        Replay
      </button>
    </div>
  );
}
