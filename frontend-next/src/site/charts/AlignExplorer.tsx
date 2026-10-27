'use client';

import { useState } from 'react';
import { alignedKeep } from '@/src/lib/align';

const WIDTHS = [16, 32, 48, 64, 128, 256];
const ALIGNS = [8, 16, 32];

/** Pick a layer width, a prune ratio and an alignment; see which channels survive. */
export function AlignExplorer() {
  const [width, setWidth] = useState(32);
  const [ratio, setRatio] = useState(0.5);
  const [align, setAlign] = useState(16);
  const r = alignedKeep(width, ratio, align);
  const cell = width > 64 ? 6 : 12;

  return (
    <div>
      <div className="grid gap-5 sm:grid-cols-3">
        <label className="block text-[13px]">
          <span className="text-(--muted)">Layer width (output channels)</span>
          <select
            value={width}
            onChange={(e) => setWidth(Number(e.target.value))}
            className="mt-1.5 block h-11 w-full rounded-xl border border-(--line-strong) bg-transparent px-3 text-[15px]"
          >
            {WIDTHS.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-[13px]">
          <span className="text-(--muted)">Requested ratio: {ratio.toFixed(2)}</span>
          <input
            type="range"
            min={0.05}
            max={0.9}
            step={0.05}
            value={ratio}
            onChange={(e) => setRatio(Number(e.target.value))}
            className="mt-3.5 block w-full accent-[var(--color-brand-600)]"
          />
        </label>
        <fieldset className="text-[13px]">
          <legend className="text-(--muted)">Requested alignment</legend>
          <div className="mt-1.5 flex gap-2">
            {ALIGNS.map((a) => (
              <button
                key={a}
                type="button"
                aria-pressed={align === a}
                onClick={() => setAlign(a)}
                className={`h-11 flex-1 rounded-xl border text-[15px] transition-colors ${
                  align === a ? 'border-transparent bg-(--inv-bg) text-(--inv-fg)' : 'border-(--line-strong) hover:bg-(--hover)'
                }`}
              >
                {a}
              </button>
            ))}
          </div>
        </fieldset>
      </div>

      <div
        className="mt-7 flex flex-wrap gap-[3px]"
        role="img"
        aria-label={`${r.kept} of ${width} channels kept`}
        style={{ maxWidth: `${Math.min(width, 64) * (cell + 3)}px` }}
      >
        {Array.from({ length: width }, (_, i) => (
          <span
            key={i}
            className={`rounded-[3px] ${i < r.kept ? 'bg-brand-600' : 'border border-(--line-strong)'}`}
            style={{ width: cell, height: cell + 8 }}
          />
        ))}
      </div>

      <dl className="figures-tab mt-6 grid gap-x-8 gap-y-2 text-[14px] sm:grid-cols-3">
        <div>
          <dt className="text-(--muted)">Kept</dt>
          <dd className="text-[26px] tracking-[-0.03em]">
            {r.kept} <span className="text-[15px] text-(--muted)">of {width}</span>
          </dd>
        </div>
        <div>
          <dt className="text-(--muted)">Actually removed</dt>
          <dd className="text-[26px] tracking-[-0.03em]">{(r.effectiveRatio * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-(--muted)">Alignment used</dt>
          <dd className="text-[26px] tracking-[-0.03em]">
            {r.alignUsed}
            {r.alignUsed !== align ? <span className="ml-2 text-[13px] text-(--muted)">shrunk from {align}</span> : null}
          </dd>
        </div>
      </dl>
    </div>
  );
}
