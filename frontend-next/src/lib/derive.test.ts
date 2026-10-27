import { describe, expect, it } from 'vitest';
import { distillation as d, quantWeights as q } from '@/src/data/results';
import { dkdTerms, fakeQuantize, histogram, softmax } from './derive';

describe('softmax from the recorded logits', () => {
  it('reproduces the recorded probabilities at T=1 and T=4', () => {
    softmax(d.teacher_logits, 1).forEach((p, i) => expect(p).toBeCloseTo(d.teacher_softmax_T1[i], 6));
    softmax(d.student_logits, 4).forEach((p, i) => expect(p).toBeCloseTo(d.student_softmax_T4[i], 6));
  });
  it('sums to one at any temperature', () => {
    for (const t of [0.5, 1, 2, 4, 8]) expect(softmax(d.teacher_logits, t).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
});

describe('decoupled KD terms', () => {
  it('reproduces the recorded TCKD and NCKD at T=4', () => {
    const { tckd, nckd } = dkdTerms(d.student_logits, d.teacher_logits, d.target, d.temperature);
    expect(tckd).toBeCloseTo(d.tckd, 4);
    expect(nckd).toBeCloseTo(d.nckd, 4);
  });
});

describe('fake quantization of the recorded channel', () => {
  const w = q.channel0.weights;
  it('uses the recorded 8-bit scale', () => {
    expect(fakeQuantize(w, 8).scale).toBeCloseTo(q.channel0.scale, 9);
  });
  it('keeps the error within half a step and shrinks it as bits grow', () => {
    const e8 = fakeQuantize(w, 8);
    expect(Math.max(...w.map((v, i) => Math.abs(v - e8.deq[i])))).toBeLessThanOrEqual(e8.scale / 2 + 1e-12);
    expect(fakeQuantize(w, 4).mse).toBeGreaterThan(e8.mse);
    expect(fakeQuantize(w, 2).levels).toBeLessThanOrEqual(3);
  });
  it('histogram conserves the count', () => {
    expect(histogram(w, -0.3, 0.3, 24).reduce((a, b) => a + b, 0)).toBe(w.length);
  });
});
