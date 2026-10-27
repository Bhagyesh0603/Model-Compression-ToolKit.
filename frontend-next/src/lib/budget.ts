import type { ModelRecord, StageMeasure } from '@/src/data/results';

/**
 * Budget checking over recorded configurations. The terms and the wording of
 * each failure mirror compresskit.search.Budget.failures; accuracy drop is
 * measured against the same model's own FP32 baseline.
 */
export type Budget = { maxLatencyMs?: number; maxDropPp?: number; maxSizeMb?: number };

export type Config = {
  id: string;
  model: string;
  arch: string;
  stage: string;
  host: string;
  kind: 'measured' | 'transcribed';
  accuracy: number;
  baselineAccuracy: number;
  median_ms: number;
  size_mb: number;
};

export const STAGE_LABELS = ['Baseline', 'Pruned', 'Recovered', 'INT8'] as const;

export function configsOf(models: ModelRecord[]): Config[] {
  return models.flatMap((m) =>
    ([m.baseline, ...m.stages] as StageMeasure[]).map((s, i) => ({
      id: `${m.arch}:${i}`,
      model: m.name,
      arch: m.arch,
      stage: STAGE_LABELS[i] ?? s.stage,
      host: m.provenance.host,
      kind: m.provenance.kind,
      accuracy: s.accuracy,
      baselineAccuracy: m.baseline.accuracy,
      median_ms: s.median_ms,
      size_mb: s.size_mb,
    })),
  );
}

export const dropPp = (c: Config) => (c.baselineAccuracy - c.accuracy) * 100;

export function failures(c: Config, b: Budget): string[] {
  const out: string[] = [];
  if (b.maxLatencyMs !== undefined && c.median_ms > b.maxLatencyMs)
    out.push(`latency ${c.median_ms.toFixed(3)} ms > ${b.maxLatencyMs.toFixed(3)} ms`);
  if (b.maxDropPp !== undefined && dropPp(c) > b.maxDropPp)
    out.push(`accuracy drop ${dropPp(c).toFixed(2)} pp > ${b.maxDropPp.toFixed(2)} pp`);
  if (b.maxSizeMb !== undefined && c.size_mb > b.maxSizeMb)
    out.push(`size ${c.size_mb.toFixed(3)} MB > ${b.maxSizeMb.toFixed(3)} MB`);
  return out;
}

/** How far over budget a configuration is, as a sum of relative excesses (0 when it meets it). */
export function excess(c: Config, b: Budget): number {
  let e = 0;
  if (b.maxLatencyMs !== undefined && c.median_ms > b.maxLatencyMs) e += c.median_ms / b.maxLatencyMs - 1;
  if (b.maxDropPp !== undefined && dropPp(c) > b.maxDropPp) e += (dropPp(c) - b.maxDropPp) / Math.max(b.maxDropPp, 0.5);
  if (b.maxSizeMb !== undefined && c.size_mb > b.maxSizeMb) e += c.size_mb / b.maxSizeMb - 1;
  return e;
}

export type Verdict = {
  meets: Config[];
  misses: { config: Config; failures: string[]; excess: number }[];
};

/** Splits the configurations on one host into those that meet the budget and those that do not. */
export function evaluate(all: Config[], b: Budget, host: string): Verdict {
  const here = all.filter((c) => c.host === host);
  const meets = here.filter((c) => failures(c, b).length === 0).sort((a, z) => a.size_mb - z.size_mb);
  const misses = here
    .filter((c) => failures(c, b).length > 0)
    .map((config) => ({ config, failures: failures(config, b), excess: excess(config, b) }))
    .sort((a, z) => a.failures.length - z.failures.length || a.excess - z.excess);
  return { meets, misses };
}
