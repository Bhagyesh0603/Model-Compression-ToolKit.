'use client';

import { motion } from 'motion/react';
import { DUR, EASE } from '@/src/motion/tokens';

/** A labelled horizontal bar that fills the first time it is on screen. `value` is 0..1 of the track. */
export function AnimBar({
  label,
  value,
  display,
  accent,
  delay = 0,
}: {
  label: string;
  value: number;
  display: string;
  accent?: boolean;
  delay?: number;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-[13px]">
        <span className="text-(--muted)">{label}</span>
        <span className="figures-tab">{display}</span>
      </div>
      <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-(--line)">
        <motion.div
          className={`h-full rounded-full ${accent ? 'bg-int8' : 'bg-(--snake)'}`}
          initial={{ width: 0 }}
          whileInView={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
          viewport={{ once: true, margin: '0px 0px -8% 0px' }}
          transition={{ duration: DUR.reveal, ease: EASE, delay }}
        />
      </div>
    </div>
  );
}
