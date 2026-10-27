type Layer = { name: string; sqnr_db: number; parameters: number };

/** SQNR per layer, worst (lowest) first. Layers marked kept stay FP32 in mixed precision. */
export function SensBars({ layers, kept, protectedLayers }: { layers: Layer[]; kept: string[]; protectedLayers: string[] }) {
  const lo = Math.min(...layers.map((l) => l.sqnr_db));
  const hi = Math.max(...layers.map((l) => l.sqnr_db));
  return (
    <ol className="figures-tab space-y-1.5" aria-label="Quantization sensitivity per layer, most sensitive first">
      {layers.map((l) => {
        const isKept = kept.includes(l.name);
        const isProtected = protectedLayers.includes(l.name);
        return (
          <li key={l.name} className="grid grid-cols-[minmax(0,10.5rem)_1fr_4.5rem] items-center gap-3 text-[12px]">
            <span className="truncate font-mono text-(--muted)">{l.name}</span>
            <span className="relative block h-2.5 rounded-full bg-(--line-strong)">
              <span
                className={`absolute inset-y-0 left-0 rounded-full ${isKept ? 'bg-(--fg)' : 'bg-brand-600'}`}
                style={{ width: `${((l.sqnr_db - lo) / (hi - lo)) * 92 + 8}%` }}
              />
            </span>
            <span className="text-right">
              {l.sqnr_db.toFixed(1)} dB
              {isKept || isProtected ? <span className="ml-1">{isKept ? '●' : '○'}</span> : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
