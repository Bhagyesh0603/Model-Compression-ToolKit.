/**
 * Pure functions that recompute, in the browser, quantities the toolkit
 * computed in Python from the same recorded inputs. Each is checked against the
 * recorded Python result in derive.test.ts, so a control that changes a
 * parameter (temperature, bit width) is exact, not a simulation.
 */

export function softmax(logits: readonly number[], temperature = 1): number[] {
  const z = logits.map((v) => v / temperature);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function klDiv(p: readonly number[], q: readonly number[]): number {
  // KL(p || q) with p the target distribution, as torch's F.kl_div(log q, p).
  return p.reduce((acc, pi, i) => (pi > 0 ? acc + pi * (Math.log(pi) - Math.log(q[i])) : acc), 0);
}

/** TCKD and NCKD of decoupled KD for one sample, as compresskit.distillation.distill.dkd_loss. */
export function dkdTerms(student: readonly number[], teacher: readonly number[], target: number, temperature: number) {
  const clamp = (v: number) => Math.min(1 - 1e-7, Math.max(1e-7, v));
  const sp = softmax(student, temperature);
  const tp = softmax(teacher, temperature);
  const bin = (p: number[]) => [clamp(p[target]), 1 - clamp(p[target])];
  const t2 = temperature * temperature;
  const tckd = klDiv(bin(tp), bin(sp)) * t2;
  const mask = (z: readonly number[]) => z.map((v, i) => v / temperature - (i === target ? 1000 : 0));
  const sMasked = softmaxRaw(mask(student));
  const tMasked = softmaxRaw(mask(teacher));
  const nckd = klDiv(tMasked, sMasked) * t2;
  return { tckd, nckd };
}

function softmaxRaw(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Symmetric per-channel fake quantization of one channel to `bits` bits:
 * scale = max|w| / (2^(bits-1) - 1), round, clamp, multiply back.
 */
export function fakeQuantize(weights: readonly number[], bits: number) {
  const qmax = 2 ** (bits - 1) - 1;
  const maxAbs = Math.max(...weights.map(Math.abs));
  const scale = maxAbs / qmax;
  const ints = weights.map((w) => Math.max(-qmax, Math.min(qmax, Math.round(w / scale))));
  const deq = ints.map((q) => q * scale);
  const mse = weights.reduce((a, w, i) => a + (w - deq[i]) ** 2, 0) / weights.length;
  return { scale, ints, deq, mse, levels: new Set(ints).size, qmax };
}

/** Counts of `values` in `bins` equal-width bins over [lo, hi]. */
export function histogram(values: readonly number[], lo: number, hi: number, bins: number): number[] {
  const out = new Array<number>(bins).fill(0);
  for (const v of values) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
    out[i] += 1;
  }
  return out;
}
