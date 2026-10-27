import { describe, expect, it } from 'vitest';
import { models } from '@/src/data/results';
import { configsOf, dropPp, evaluate, failures } from './budget';

const all = configsOf(models);
const m5 = 'Apple M5';

describe('budget check over recorded configurations', () => {
  it('lists four configurations per model', () => {
    expect(all).toHaveLength(models.length * 4);
  });

  it('measures accuracy drop against the model’s own baseline', () => {
    const r20int8 = all.find((c) => c.id === 'resnet20:3')!;
    expect(dropPp(r20int8)).toBeCloseTo(7.54, 2);
  });

  it('reports failures in the toolkit’s wording', () => {
    const r20int8 = all.find((c) => c.id === 'resnet20:3')!;
    expect(failures(r20int8, { maxLatencyMs: 0.3 })).toEqual(['latency 0.460 ms > 0.300 ms']);
    expect(failures(r20int8, { maxLatencyMs: 0.5, maxDropPp: 8 })).toEqual([]);
  });

  it('only compares configurations measured on the same host', () => {
    const v = evaluate(all, { maxLatencyMs: 100 }, m5);
    expect(v.meets.every((c) => c.host === m5)).toBe(true);
    expect(v.meets.length + v.misses.length).toBe(all.filter((c) => c.host === m5).length);
  });

  it('puts the closest miss first when nothing meets the budget', () => {
    const v = evaluate(all, { maxLatencyMs: 0.01, maxDropPp: 0.1 }, m5);
    expect(v.meets).toHaveLength(0);
    expect(v.misses[0].failures.length).toBeLessThanOrEqual(v.misses[v.misses.length - 1].failures.length);
  });
});
