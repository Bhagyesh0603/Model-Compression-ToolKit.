'use client';

import { useState } from 'react';

type Props = {
  classes: string[];
  target: number;
  teacherT1: number[];
  studentT1: number[];
  teacherT4: number[];
  studentT4: number[];
  temperature: number;
};

/** Teacher and student class probabilities for one image, at T=1 and at the distillation temperature. */
export function ClassBars({ classes, target, teacherT1, studentT1, teacherT4, studentT4, temperature }: Props) {
  const [soft, setSoft] = useState(false);
  const teacher = soft ? teacherT4 : teacherT1;
  const student = soft ? studentT4 : studentT1;

  return (
    <div>
      <div className="mb-5 inline-flex rounded-full border border-(--line-strong) p-1 text-[13px]" role="group" aria-label="Temperature">
        {[
          [false, 'T = 1'],
          [true, `T = ${temperature}`],
        ].map(([v, label]) => (
          <button
            key={label as string}
            type="button"
            aria-pressed={soft === v}
            onClick={() => setSoft(v as boolean)}
            className={`h-9 rounded-full px-4 transition-colors ${soft === v ? 'bg-(--inv-bg) text-(--inv-fg)' : 'hover:bg-(--hover)'}`}
          >
            {label as string}
          </button>
        ))}
      </div>
      <ol className="figures-tab space-y-2">
        {classes.map((name, i) => (
          <li key={name} className="grid grid-cols-[6.5rem_1fr_7rem] items-center gap-3 text-[12px]">
            <span className={i === target ? 'font-semibold' : 'text-(--muted)'}>
              {name}
              {i === target ? ' (true)' : ''}
            </span>
            <span className="block space-y-1">
              <span className="block h-2 rounded-full bg-(--chip)">
                <span className="block h-full rounded-full bg-(--fg)" style={{ width: `${teacher[i] * 100}%` }} />
              </span>
              <span className="block h-2 rounded-full bg-(--chip)">
                <span className="block h-full rounded-full bg-brand-600" style={{ width: `${student[i] * 100}%` }} />
              </span>
            </span>
            <span className="text-right text-(--muted)">
              {(teacher[i] * 100).toFixed(1)}% / {(student[i] * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ol>
      <p className="mt-4 flex gap-5 text-[12px] text-(--muted)">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-4 rounded-full bg-(--fg)" /> teacher
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-4 rounded-full bg-brand-600" /> student
        </span>
      </p>
    </div>
  );
}
