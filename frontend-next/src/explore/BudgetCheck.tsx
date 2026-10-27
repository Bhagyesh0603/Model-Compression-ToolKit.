'use client';

import { useMemo, useState } from 'react';
import { Check, X } from 'lucide-react';
import { motion } from 'motion/react';
import { models, ms, pct, mb } from '@/src/data/results';
import { configsOf, dropPp, evaluate, type Budget, type Config } from '@/src/lib/budget';
import { DUR, EASE } from '@/src/motion/tokens';

const ALL = configsOf(models);
const HOSTS = [...new Set(models.map((m) => m.provenance.host))];

const PRESETS: { label: string; latency: string; drop: string; size: string }[] = [
  { label: 'Fast and nearly as accurate', latency: '0.5', drop: '8', size: '' },
  { label: 'Tiny file, almost no loss', latency: '', drop: '1', size: '0.3' },
  { label: 'Impossibly strict', latency: '0.02', drop: '0.5', size: '' },
];

const num = (s: string) => {
  const v = parseFloat(s);
  return Number.isFinite(v) && v > 0 ? v : undefined;
};

/**
 * Type a budget and see which recorded configurations meet it. This is not a
 * search: it filters what was already measured, on one host at a time, so
 * latencies from different machines are never compared.
 */
export function BudgetCheck() {
  const [host, setHost] = useState(HOSTS[0]);
  const [latency, setLatency] = useState('0.5');
  const [drop, setDrop] = useState('8');
  const [size, setSize] = useState('');
  const budget: Budget = { maxLatencyMs: num(latency), maxDropPp: num(drop), maxSizeMb: num(size) };
  const empty = budget.maxLatencyMs === undefined && budget.maxDropPp === undefined && budget.maxSizeMb === undefined;
  const verdict = useMemo(() => evaluate(ALL, budget, host), [host, latency, drop, size]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)] lg:gap-14">
      <div className="space-y-6">
        <fieldset>
          <legend className="label-xs !text-(--muted)">Measured on</legend>
          <div className="mt-3 flex flex-wrap gap-2">
            {HOSTS.map((h) => (
              <button
                key={h}
                type="button"
                aria-pressed={host === h}
                onClick={() => setHost(h)}
                className="h-9 rounded-full border border-(--line-strong) px-4 text-[13px] transition-colors aria-pressed:border-transparent aria-pressed:bg-(--inv-bg) aria-pressed:text-(--inv-fg) hover:bg-(--hover) aria-pressed:hover:bg-(--inv-bg)"
              >
                {h.replace(/^Kaggle \(.*\)$/, 'Kaggle CPU')}
              </button>
            ))}
          </div>
        </fieldset>
        <div className="grid grid-cols-3 gap-3">
          <Field id="b-lat" label="Max latency, ms" value={latency} onChange={setLatency} />
          <Field id="b-drop" label="Max drop, pp" value={drop} onChange={setDrop} />
          <Field id="b-size" label="Max size, MB" value={size} onChange={setSize} />
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => {
                setLatency(p.latency);
                setDrop(p.drop);
                setSize(p.size);
              }}
              className="rounded-full bg-(--chip) px-3.5 py-1.5 text-[13px] text-(--chip-fg) transition-colors hover:bg-(--chip-hover)"
            >
              {p.label}
            </button>
          ))}
        </div>
        <p className="text-[13px] leading-[1.6] text-(--muted)">
          Leave a field empty to ignore that term. The wording of each failure matches the toolkit&rsquo;s own budget search.
          Drop is measured against that model&rsquo;s own baseline.
        </p>
      </div>

      <div aria-live="polite">
        {empty ? (
          <p className="text-[15px] text-(--muted)">Enter at least one limit.</p>
        ) : (
          <>
            <p className="text-[clamp(20px,2.2vw,30px)] leading-[1.15] tracking-[-0.03em]">
              {verdict.meets.length > 0
                ? `${verdict.meets.length} recorded configuration${verdict.meets.length === 1 ? '' : 's'} meet this budget.`
                : 'No recorded configuration meets this budget.'}
            </p>
            {verdict.meets.length === 0 && verdict.misses[0] ? (
              <p className="mt-2 text-[14px] text-(--muted)">
                The closest is {verdict.misses[0].config.model}, {verdict.misses[0].config.stage}: {verdict.misses[0].failures.join('; ')}.
                The toolkit&rsquo;s search would try more compression and say so if it still fell short.
              </p>
            ) : null}
            <ul className="mt-6 divide-y divide-(--line) border-y border-(--line)">
              {verdict.meets.map((c) => (
                <Row key={c.id} c={c} ok />
              ))}
              {verdict.misses.map((m) => (
                <Row key={m.config.id} c={m.config} reasons={m.failures} />
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

function Field({ id, label, value, onChange }: { id: string; label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <label htmlFor={id} className="label-xs !text-(--muted)">
        {label}
      </label>
      <input
        id={id}
        inputMode="decimal"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="figures-tab mt-2 h-11 w-full rounded-xl border border-(--line-strong) bg-transparent px-3 text-[15px] outline-none transition-colors focus:border-(--snake)"
      />
    </div>
  );
}

function Row({ c, ok, reasons }: { c: Config; ok?: boolean; reasons?: string[] }) {
  return (
    <motion.li
      layout
      transition={{ duration: DUR.layout, ease: EASE }}
      className={`flex items-start gap-3 py-3.5 ${ok ? '' : 'opacity-60'}`}
    >
      <span
        className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full ${ok ? 'bg-(--snake) text-(--bg)' : 'bg-(--chip) text-(--chip-fg)'}`}
        aria-label={ok ? 'meets the budget' : 'misses the budget'}
      >
        {ok ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <X className="h-3.5 w-3.5" aria-hidden="true" />}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[15px]">
          {c.model} <span className="text-(--muted)">· {c.stage}</span>
        </p>
        <p className="figures-tab mt-0.5 text-[13px] text-(--muted)">
          {ms(c.median_ms)} · {mb(c.size_mb)} · {pct(c.accuracy)} ({dropPp(c).toFixed(2)} pp {dropPp(c) > 0 ? 'lower' : 'change'})
        </p>
        {reasons ? <p className="mt-0.5 text-[12px] text-(--muted)">{reasons.join('; ')}</p> : null}
      </div>
    </motion.li>
  );
}
