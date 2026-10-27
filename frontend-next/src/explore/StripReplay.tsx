'use client';

import { useEffect, useRef } from 'react';
import { RotateCcw } from 'lucide-react';
import { animate, useInView, useMotionValue } from 'motion/react';
import { models } from '@/src/data/results';
import { usePrefersReducedMotion } from '@/src/landing/hooks';
import { EASE } from '@/src/motion/tokens';
import { ModelStrip } from '@/src/viz/ModelStrip';
import { layersOf } from '@/src/viz/layers';


/** The recorded channel cut of ResNet-20, played when it scrolls into view. */
export function StripReplay({ arch = 'resnet20' }: { arch?: string }) {
  const model = models.find((m) => m.arch === arch)!;
  const layers = layersOf(model);
  const ref = useRef<HTMLDivElement>(null);
  const seen = useInView(ref, { once: true, margin: '0px 0px -20% 0px' });
  const reduce = usePrefersReducedMotion();
  const cut = useMotionValue(reduce ? 1 : 0);
  const quant = useMotionValue(0);
  const play = () => {
    cut.set(0);
    return animate(cut, 1, { duration: 2, ease: EASE });
  };
  useEffect(() => {
    if (seen && !reduce) {
      const c = play();
      return () => c.stop();
    }
  }, [seen, reduce]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!layers) return null;
  return (
    <div ref={ref}>
      <div className="mb-4 flex items-center justify-between gap-3 text-[12px] text-(--muted)">
        <ul className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <li className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-[3px] bg-(--line-strong)" />removed</li>
          <li className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-[3px] bg-(--snake)" />kept</li>
        </ul>
        <button type="button" onClick={() => play()} className="inline-flex h-8 items-center gap-1.5 rounded-full bg-(--chip) px-3 text-(--chip-fg) transition-colors hover:bg-(--chip-hover)">
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          Replay
        </button>
      </div>
      <ModelStrip layers={layers ?? []} cut={cut} quant={quant} label={`${model.name} channels per layer, before and after the recorded prune.`} className="h-[clamp(140px,22vw,260px)] w-full" />
    </div>
  );
}
