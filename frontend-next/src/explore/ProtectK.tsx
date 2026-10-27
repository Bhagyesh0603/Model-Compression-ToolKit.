'use client';

import { useState } from 'react';
import { sensitivity as s } from '@/src/data/results';

/** Pick how many of the most sensitive layers to hold in FP32; the ranking is the recorded SQNR. */
export function ProtectK() {
  const [k, setK] = useState(2);
  const layers = [...s.layers].sort((a, b) => a.sqnr_db - b.sqnr_db);
  const lo = Math.floor(layers[0].sqnr_db / 5) * 5;
  const hi = Math.ceil(layers[layers.length - 1].sqnr_db / 5) * 5;
  const kept = layers.slice(0, k);

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] lg:gap-14">
      <ul aria-label="Layers by quantization sensitivity, most sensitive first" className="space-y-1.5">
        {layers.map((l, i) => (
          <li key={l.name} className="grid grid-cols-[168px_1fr_52px] items-center gap-3 text-[12px]">
            <span className={`truncate font-mono ${i < k ? 'text-(--fg)' : 'text-(--muted)'}`}>{l.name}</span>
            <span className="h-2 overflow-hidden rounded-full bg-(--line)">
              <span
                className={`block h-full rounded-full transition-colors duration-500 ${i < k ? 'bg-int8' : 'bg-(--snake)'}`}
                style={{ width: `${((l.sqnr_db - lo) / (hi - lo)) * 100}%` }}
              />
            </span>
            <span className="figures-tab text-right text-(--muted)">{l.sqnr_db.toFixed(1)} dB</span>
          </li>
        ))}
      </ul>
      <div>
        <div className="flex items-center gap-4">
          <label htmlFor="protect" className="label-xs !text-(--muted)">
            Hold in FP32
          </label>
          <input id="protect" type="range" min={0} max={6} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} className="w-full max-w-[200px] accent-[var(--snake)]" />
          <output className="figures-tab w-8 text-[15px]">{k}</output>
        </div>
        <p className="mt-5 min-h-[88px] text-[14px] leading-[1.65]">
          {k === 0 ? (
            'Every layer is quantized.'
          ) : (
            <>
              Kept in float:{' '}
              {kept.map((l, i) => (
                <span key={l.name}>
                  {i ? ', ' : ''}
                  <span className="font-mono text-[13px]">{l.name}</span>
                </span>
              ))}
              .
            </>
          )}
        </p>
        <p className="text-[13px] leading-[1.6] text-(--muted)">
          SQNR is the signal-to-noise ratio at the logits when only that layer&rsquo;s weights take an INT8 round trip,
          measured on {s.batches} batches of training images with no labels. Lower means more damage. The toolkit holds the
          {' '}two lowest in float by default; the accuracy effect of holding more was not measured here.
        </p>
      </div>
    </div>
  );
}
