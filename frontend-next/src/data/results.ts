/**
 * Typed access to the JSON written by tools/export_site_data.py.
 *
 * Nothing on the site states a measurement that isn't read from here. The JSON
 * files are generated, not edited by hand (the one hand-typed source is
 * tools/transcribed_vgg16.json, and its records say so in `provenance.kind`).
 */
import modelsJson from './models.json';
import hostsJson from './hosts.json';
import sensitivityJson from './sensitivity.json';
import quantJson from './quant_weights.json';
import bnJson from './bn_recal.json';
import distillJson from './distillation.json';
import alignJson from './align_golden.json';
import gradualJson from './gradual_vs_oneshot.json';
import searchJson from './search.json';
import notebooksJson from './notebooks.json';
import verifyJson from './verify.json';

export type ProvenanceKind = 'measured' | 'transcribed';

export interface Provenance {
  kind: ProvenanceKind;
  host: string;
  where: string;
  note?: string;
  date?: string;
}

export interface StageMeasure {
  stage: string;
  precision: string;
  accuracy: number;
  median_ms: number;
  p95_ms: number;
  size_mb: number;
  parameters: number;
  notes?: string;
  recall?: number[] | null;
  support?: number[] | null;
  artifact: string;
}

export interface PruneStep {
  step: number;
  params: number | null;
  acc_after_cut: number;
  acc_after_finetune: number;
}

export interface ModelRecord {
  name: string;
  arch: string;
  dataset: string;
  schedule: string;
  source: string;
  provenance: Provenance;
  host: { cpu: string; machine: string; int8_speed: string };
  baseline: StageMeasure;
  stages: StageMeasure[];
  prune_stats: {
    params_before: number;
    params_after: number;
    macs_before: number;
    macs_after: number;
    ratio_requested: number;
    steps: number;
    channels: Record<string, [number, number]>;
  } | null;
  prune_history: PruneStep[];
  quantization: { precision: string; coverage: number; quantized_ops: Record<string, number> } | null;
  totals: { accuracy_drop_pp: number; speedup: number; size_reduction: number; efficiency_score: number };
  warnings: string[];
  downloads?: { label: string; href: string; size_bytes: number }[];
}

export const models = modelsJson.models as unknown as ModelRecord[];
export const hosts = hostsJson.hosts;
export const sensitivity = sensitivityJson;
export const quantWeights = quantJson;
export const bnRecal = bnJson;
export const distillation = distillJson;
export const alignGolden = alignJson;
export const gradualVsOneShot = gradualJson;
export const search = searchJson;
export const notebooks = notebooksJson.notebooks;
export const verify = verifyJson;

export function slugOf(arch: string): string {
  return arch.replace(/_/g, '-');
}

export function modelBySlug(slug: string): ModelRecord | undefined {
  return models.find((m) => slugOf(m.arch) === slug);
}

export function finalStage(m: ModelRecord): StageMeasure {
  return m.stages[m.stages.length - 1];
}

/** 0.9154 -> "91.54%" */
export const pct = (fraction: number, digits = 2) => `${(fraction * 100).toFixed(digits)}%`;
/** 9.11 -> "9.1x" */
export const ratio = (x: number, digits = 1) => `${x.toFixed(digits)}x`;
export const ms = (v: number) => `${v.toFixed(v < 1 ? 3 : 2)} ms`;
export const mb = (v: number) => `${v.toFixed(v < 10 ? 2 : 1)} MB`;
export const count = (n: number) => n.toLocaleString('en-US');
