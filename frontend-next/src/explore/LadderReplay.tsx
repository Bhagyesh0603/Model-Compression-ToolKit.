'use client';

import { useState } from 'react';
import { Check, RotateCcw, X } from 'lucide-react';
import { motion } from 'motion/react';
import { mb, ms, pct, search } from '@/src/data/results';
import { DUR, EASE } from '@/src/motion/tokens';

/** Replay the escalation ladder the budget search walked for each recorded budget. */
export function LadderReplay() {
  const [run, setRun] = useState(0);
  const [replay, setReplay] = useState(0);
  const r = search.runs[run];

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div role="tablist" aria-label="Recorded budgets" className="flex flex-wrap gap-2">
          {search.runs.map((x, i) => (
            <button
              key={x.budget}
              role="tab"
              type="button"
              aria-selected={i === run}
              onClick={() => {
                setRun(i);
                setReplay((n) => n + 1);
              }}
              className="h-9 rounded-full border border-(--line-strong) px-4 text-[13px] transition-colors aria-selected:border-transparent aria-selected:bg-(--inv-bg) aria-selected:text-(--inv-fg) hover:bg-(--hover) aria-selected:hover:bg-(--inv-bg)"
            >
              {x.described}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setReplay((n) => n + 1)}
          className="inline-flex h-9 items-center gap-1.5 rounded-full bg-(--chip) px-3.5 text-[13px] text-(--chip-fg) transition-colors hover:bg-(--chip-hover)"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          Replay
        </button>
      </div>

      <ol key={`${run}-${replay}`} className="mt-8 space-y-3">
        {r.attempts.map((a, i) => (
          <motion.li
            key={a.label}
            initial={{ opacity: 0, x: -14 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: DUR.enter, ease: EASE, delay: i * 0.45 }}
            className="grid grid-cols-[28px_1fr] gap-3 rounded-2xl border border-(--line) p-4 sm:grid-cols-[28px_1.2fr_1fr]"
          >
            <span className={`mt-0.5 flex h-6 w-6 items-center justify-center rounded-full ${a.met ? 'bg-(--snake) text-(--bg)' : 'bg-(--chip) text-(--chip-fg)'}`}>
              {a.met ? <Check className="h-3.5 w-3.5" aria-label="meets the budget" /> : <X className="h-3.5 w-3.5" aria-label="misses the budget" />}
            </span>
            <span>
              <span className="block text-[15px]">{a.label}</span>
              <span className="mt-0.5 block text-[12px] leading-snug text-(--muted)">{a.plan}</span>
            </span>
            <span className="figures-tab col-span-2 text-[13px] text-(--muted) sm:col-span-1 sm:text-right">
              {pct(a.accuracy)} · {ms(a.median_ms)} · {mb(a.size_mb)}
              {a.failures.length ? <span className="mt-0.5 block text-[12px]">{a.failures.join('; ')}</span> : null}
            </span>
          </motion.li>
        ))}
      </ol>
      <motion.p
        key={`v-${run}-${replay}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: DUR.enter, ease: EASE, delay: r.attempts.length * 0.45 }}
        className="mt-6 text-[16px] leading-[1.5]"
      >
        {r.met
          ? `Met by ${r.chosen}.`
          : `Not met. The search returns its best trade-off, ${r.chosen}, and says the budget was missed.`}
      </motion.p>
    </div>
  );
}
