import { describe, expect, it } from 'vitest';
import golden from '../data/align_golden.json';
import { alignedKeep, pyRound } from './align';

describe('alignedKeep matches the Python implementation', () => {
  it('replays every recorded case exactly', () => {
    expect(golden.rows.length).toBeGreaterThan(100);
    for (const row of golden.rows) {
      const got = alignedKeep(row.width, row.ratio, row.align);
      const label = `width ${row.width}, ratio ${row.ratio}, align ${row.align}`;
      expect(got.kept, label).toBe(row.kept);
      expect(got.alignUsed, label).toBe(row.align_used);
      expect(got.effectiveRatio, label).toBeCloseTo(row.effective_ratio, 12);
    }
  });

  it('rounds halves to even like Python', () => {
    expect([pyRound(0.5), pyRound(1.5), pyRound(2.5), pyRound(3.5)]).toEqual([0, 2, 2, 4]);
  });
});
