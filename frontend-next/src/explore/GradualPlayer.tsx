'use client';

import { useRef } from 'react';
import { Play } from 'lucide-react';
import { animate, useMotionValue, useMotionValueEvent } from 'motion/react';
import { gradualVsOneShot as g, pct } from '@/src/data/results';
import { EASE } from '@/src/motion/tokens';
import { AnimBar } from '@/src/viz/AnimBar';
import { AnimSawtooth } from '@/src/viz/AnimSawtooth';

/** Draw the recorded gradual run step by step, with the one-shot run beside it at the same fine-tune budget. */
export function GradualPlayer() {
  const reveal = useMotionValue(1);
  const slider = useRef<HTMLInputElement>(null);
  useMotionValueEvent(reveal, 'change', (v) => {
    if (slider.current) slider.current.value = String(v);
  });
  const play = () => {
    reveal.set(0);
    animate(reveal, 1, { duration: 3.4, ease: EASE });
  };
  const gFinal = g.gradual.history[g.gradual.history.length - 1].acc_after_finetune;
  const oShot = g.one_shot.history[0];

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:gap-14">
      <div>
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={play}
            className="inline-flex h-9 items-center gap-2 rounded-full bg-(--inv-bg) px-4 text-[13px] font-medium text-(--inv-fg) transition-colors hover:bg-(--inv-hover)"
          >
            <Play className="h-3.5 w-3.5" aria-hidden="true" />
            Play the cuts
          </button>
          <label htmlFor="reveal" className="sr-only">
            Progress through the pruning steps
          </label>
          <input
            id="reveal"
            ref={slider}
            type="range"
            min={0}
            max={1}
            step={0.01}
            defaultValue={1}
            onChange={(e) => reveal.set(Number(e.target.value))}
            className="w-full max-w-[280px] accent-[var(--snake)]"
          />
        </div>
        <AnimSawtooth
          baseline={g.baseline_accuracy}
          history={g.gradual.history}
          reveal={reveal}
          summary="Accuracy after each of three cuts and after each fine-tune on ResNet-20."
          className="mt-4 w-full"
        />
        <p className="mt-2 text-[12px] text-(--muted)">
          {g.gradual.steps} cuts of {g.gradual.epochs_per_step} fine-tune epochs each. Open dot: right after a cut. Solid:
          after fine-tuning.
        </p>
      </div>
      <div className="space-y-5">
        <p className="label-xs !text-(--muted)">Same {g.total_fine_tune_epochs} fine-tune epochs, one cut instead</p>
        <AnimBar label={`One-shot, after the cut`} value={oShot.acc_after_cut} display={pct(oShot.acc_after_cut)} />
        <AnimBar label={`One-shot, after ${g.one_shot.epochs_per_step} epochs`} value={oShot.acc_after_finetune} display={pct(oShot.acc_after_finetune)} accent delay={0.1} />
        <AnimBar label="Gradual, final" value={gFinal} display={pct(gFinal)} delay={0.2} />
        <p className="text-[13px] leading-[1.6] text-(--muted)">
          One run each. One-shot finished higher here, so gradual pruning is not shown to be better at an equal budget on
          this model, and the toolkit does not claim it is.
        </p>
      </div>
    </div>
  );
}
