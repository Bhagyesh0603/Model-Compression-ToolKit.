import type { Provenance as P } from '@/src/data/results';

/** Where a figure came from. Printed under every chart and table, never omitted. */
export function Provenance({ p, source }: { p: P; source?: string }) {
  return (
    <p className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] leading-snug text-(--muted)">
      <span
        className={`inline-flex h-5 items-center rounded-full px-2 text-[10px] font-semibold uppercase tracking-[0.06em] ${
          p.kind === 'measured' ? 'bg-brand-600 text-white' : 'border border-(--line-strong) text-(--fg)'
        }`}
      >
        {p.kind}
      </span>
      <span>{p.host}</span>
      <span aria-hidden="true">·</span>
      <span>{p.where}</span>
      {source ? (
        <>
          <span aria-hidden="true">·</span>
          <span className="font-mono">{source}</span>
        </>
      ) : null}
      {p.note ? <span className="basis-full">{p.note}</span> : null}
    </p>
  );
}
