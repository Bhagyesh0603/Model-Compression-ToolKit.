'use client';

import { useMemo, useState } from 'react';
import { distillation as d } from '@/src/data/results';
import { dkdTerms, softmax } from '@/src/lib/derive';

/** Temperature slider over the recorded teacher and student logits for one real test image. */
export function TempExplorer() {
  const [t, setT] = useState<number>(d.temperature);
  const teacher = useMemo(() => softmax(d.teacher_logits, t), [t]);
  const student = useMemo(() => softmax(d.student_logits, t), [t]);
  const { tckd, nckd } = useMemo(() => dkdTerms(d.student_logits, d.teacher_logits, d.target, t), [t]);
  const total = d.alpha * tckd + d.beta * nckd;

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:gap-14">
      <div>
        <div className="flex items-center gap-4">
          <label htmlFor="temp" className="label-xs !text-(--muted)">
            Temperature
          </label>
          <input
            id="temp"
            type="range"
            min={1}
            max={8}
            step={0.5}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
            className="w-full max-w-[320px] accent-[var(--snake)]"
          />
          <output className="figures-tab w-12 text-[15px]">T = {t}</output>
        </div>
        <ul className="mt-6 space-y-2.5" aria-label={`Teacher and student probability per class at T = ${t}`}>
          {d.classes.map((c, i) => (
            <li key={c} className="grid grid-cols-[84px_1fr_auto] items-center gap-3 text-[13px]">
              <span className={i === d.target ? 'font-medium' : 'text-(--muted)'}>
                {c}
                {i === d.target ? ' (true)' : ''}
              </span>
              <span className="space-y-1">
                <Bar value={teacher[i]} className="bg-(--snake)" />
                <Bar value={student[i]} className="bg-int8" />
              </span>
              <span className="figures-tab w-[88px] text-right text-[12px] text-(--muted)">
                {(teacher[i] * 100).toFixed(1)} / {(student[i] * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 flex gap-5 text-[12px] text-(--muted)">
          <span className="flex items-center gap-2"><span className="h-2 w-4 rounded-full bg-(--snake)" />teacher</span>
          <span className="flex items-center gap-2"><span className="h-2 w-4 rounded-full bg-int8" />student</span>
        </p>
      </div>

      <div>
        <p className="label-xs !text-(--muted)">Decoupled KD at T = {t}</p>
        <table className="figures-tab mt-3 w-full text-left text-[14px]">
          <tbody>
            <tr className="border-t border-(--line)">
              <td className="py-3">TCKD, target vs the rest</td>
              <td className="py-3 text-right">{tckd.toFixed(3)}</td>
            </tr>
            <tr className="border-t border-(--line)">
              <td className="py-3">NCKD, ranking of the other classes</td>
              <td className="py-3 text-right">{nckd.toFixed(3)}</td>
            </tr>
            <tr className="border-t border-(--line) font-medium">
              <td className="py-3">
                {d.alpha} × TCKD + {d.beta} × NCKD
              </td>
              <td className="py-3 text-right">{total.toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-4 text-[13px] leading-[1.6] text-(--muted)">
          Computed here from the recorded logits with the same formula as the toolkit&rsquo;s loss, and checked against the
          recorded result at T = {d.temperature}. A higher temperature flattens both distributions, so the ranking among
          wrong classes becomes visible.
        </p>
      </div>
    </div>
  );
}

function Bar({ value, className }: { value: number; className: string }) {
  return (
    <span className="block h-1.5 overflow-hidden rounded-full bg-(--line)">
      <span
        className={`block h-full rounded-full transition-[width] duration-500 ease-[cubic-bezier(.2,.7,.2,1)] ${className}`}
        style={{ width: `${Math.max(0.5, value * 100)}%` }}
      />
    </span>
  );
}
