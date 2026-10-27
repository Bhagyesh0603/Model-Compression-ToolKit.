import type { ModelRecord } from '@/src/data/results';

export type StripLayer = { name: string; total: number; kept: number };

/** Per-layer channel counts from a run's prune stats, or null when the run did not record them. */
export function layersOf(model: ModelRecord): StripLayer[] | null {
  const channels = model.prune_stats?.channels;
  if (!channels) return null;
  const layers = Object.entries(channels).map(([name, [total, kept]]) => ({ name, total, kept }));
  return layers.length > 0 ? layers : null;
}
