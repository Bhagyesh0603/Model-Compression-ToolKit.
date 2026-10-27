/** The technique pages, in reading order. One route each: /techniques/<slug>. */
export const TECHNIQUES = [
  { slug: 'pruning', label: 'Structured pruning', short: 'Which channels go, and how widths stay SIMD-friendly' },
  { slug: 'gradual', label: 'Gradual pruning', short: 'Cut in steps, fine-tune after each' },
  { slug: 'recalibration', label: 'BatchNorm recalibration', short: 'Re-estimating statistics after a cut' },
  { slug: 'distillation', label: 'Distillation', short: 'The original model as teacher' },
  { slug: 'quantization', label: 'Quantization', short: 'INT8, FP16 and QAT' },
  { slug: 'sensitivity', label: 'Sensitivity', short: 'Which layers quantization hurts most' },
  { slug: 'search', label: 'Budget search', short: 'State a budget, find the least compression' },
  { slug: 'benchmarking', label: 'Benchmarking', short: 'How every number is measured' },
] as const;

export type TechniqueSlug = (typeof TECHNIQUES)[number]['slug'];
