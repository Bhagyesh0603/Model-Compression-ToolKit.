type Row = { name: string; before: number; after: number };

/**
 * Channels kept per layer. The pale bar is the original width, the solid bar is
 * what survived; a layer whose kept count is not a multiple of `align` is marked,
 * because that is where the SIMD alignment had to shrink.
 */
export function ChannelBars({ rows, align }: { rows: Row[]; align: number }) {
  const widest = Math.max(...rows.map((r) => r.before));
  return (
    <ol className="figures-tab space-y-1.5" aria-label="Channels kept per layer">
      {rows.map((r) => {
        const off = r.after % align !== 0;
        return (
          <li key={r.name} className="grid grid-cols-[minmax(0,9rem)_1fr_4.5rem] items-center gap-3 text-[12px]">
            <span className="truncate font-mono text-(--muted)">{r.name}</span>
            <span className="relative block h-2.5 rounded-full bg-(--line-strong)" style={{ width: `${(r.before / widest) * 100}%` }}>
              <span
                className="absolute inset-y-0 left-0 rounded-full bg-brand-600"
                style={{ width: `${(r.after / r.before) * 100}%` }}
              />
            </span>
            <span className="text-right">
              {r.before} to {r.after}
              {off ? (
                <span className="ml-1 text-(--muted)" title={`not a multiple of ${align}`}>
                  *
                </span>
              ) : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
