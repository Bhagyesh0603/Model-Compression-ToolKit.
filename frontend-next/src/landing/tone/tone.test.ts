import { describe, expect, it } from 'vitest';
import {
  TONE,
  bandIndexAt,
  lineY,
  scrollFloor,
  selectBand,
  stepMix,
  type Selection,
  type ToneBand,
} from './tone';

const VH = 900;
const BANDS: ToneBand[] = [
  { id: 'hero', tone: 'light', top: 0, bottom: 2600 },
  { id: 'how', tone: 'dark', top: 2600, bottom: 5840 },
  { id: 'report-types', tone: 'light', top: 5840, bottom: 6940 },
  { id: 'promises', tone: 'dark', top: 6940, bottom: 7940 },
  { id: 'closing', tone: 'light', top: 7940, bottom: 8640 },
  { id: 'site-footer', tone: 'dark', top: 8640, bottom: 9340 },
];
const MAX_SCROLL = 9340 - VH;
const FRAME = 1000 / 60;

describe('lineY', () => {
  it('sits at 60% of the viewport away from the bottom of the page', () => {
    expect(lineY(0, VH, MAX_SCROLL)).toBe(540);
    expect(lineY(3000, VH, MAX_SCROLL)).toBe(3540);
  });

  it('slides to the viewport bottom over the last viewport of scroll so the footer is reachable', () => {
    expect(lineY(MAX_SCROLL, VH, MAX_SCROLL)).toBe(9340);
    expect(bandIndexAt(BANDS, lineY(MAX_SCROLL, VH, MAX_SCROLL))).toBe(5);
  });

  it('is monotone in scrollY', () => {
    let prev = -Infinity;
    for (let s = 0; s <= MAX_SCROLL; s += 7) {
      const y = lineY(s, VH, MAX_SCROLL);
      expect(y).toBeGreaterThanOrEqual(prev);
      prev = y;
    }
  });
});

describe('bandIndexAt', () => {
  it('finds the band containing y and clamps outside the range', () => {
    expect(bandIndexAt(BANDS, -10)).toBe(0);
    expect(bandIndexAt(BANDS, 2599.9)).toBe(0);
    expect(bandIndexAt(BANDS, 2600)).toBe(1);
    expect(bandIndexAt(BANDS, 99999)).toBe(5);
    expect(bandIndexAt([], 10)).toBe(-1);
  });
});

describe('selectBand', () => {
  it('takes the first selection as a jump', () => {
    expect(selectBand(BANDS, 100, null)).toEqual({ index: 0, entry: 'jump' });
  });

  it('holds the current band until the line is 48px past the edge, in both directions', () => {
    let sel = selectBand(BANDS, 100, null);
    sel = selectBand(BANDS, 2600 + 47, sel);
    expect(sel.index).toBe(0);
    sel = selectBand(BANDS, 2600 + 48, sel);
    expect(sel).toEqual({ index: 1, entry: 'above' });
    sel = selectBand(BANDS, 2600 - 48, sel);
    expect(sel.index).toBe(1);
    sel = selectBand(BANDS, 2600 - 49, sel);
    expect(sel).toEqual({ index: 0, entry: 'below' });
  });

  it('treats a non-adjacent band as a jump (anchor link, resize)', () => {
    expect(selectBand(BANDS, 7000, { index: 0, entry: 'jump' })).toEqual({ index: 3, entry: 'jump' });
  });

  it('never flaps while the line oscillates within the hysteresis window', () => {
    let sel: Selection = { index: 1, entry: 'above' };
    const changes: number[] = [];
    for (const offset of [-10, 30, -40, 47, -47, 20, 0, 47]) {
      const next = selectBand(BANDS, 5840 + offset, sel);
      if (next.index !== sel.index) changes.push(next.index);
      sel = next;
    }
    expect(changes).toEqual([]);
  });
});

describe('scrollFloor', () => {
  const enteredHowFromAbove: Selection = { index: 1, entry: 'above' };

  it('is 0 at the moment of switching and 1 before the pinned section pins', () => {
    const atSwitch = 2600 - (TONE.switchLine * VH - TONE.hysteresisPx);
    expect(scrollFloor(BANDS, enteredHowFromAbove, atSwitch, VH)).toBeCloseTo(0, 9);
    expect(scrollFloor(BANDS, enteredHowFromAbove, 2600 - TONE.completeDown * VH, VH)).toBe(1);
    expect(scrollFloor(BANDS, enteredHowFromAbove, 2600, VH)).toBe(1);
    const halfway = scrollFloor(BANDS, enteredHowFromAbove, 2600 - 0.4 * VH, VH);
    expect(halfway).toBeGreaterThan(0);
    expect(halfway).toBeLessThan(1);
  });

  it('scrolling up, is complete before the pinned panel fills the screen', () => {
    const enteredHowFromBelow: Selection = { index: 1, entry: 'below' };
    expect(scrollFloor(BANDS, enteredHowFromBelow, 5840 - (TONE.switchLine * VH + TONE.hysteresisPx), VH)).toBeCloseTo(0, 9);
    expect(scrollFloor(BANDS, enteredHowFromBelow, 5840 - TONE.completeUp * VH, VH)).toBe(1);
    expect(scrollFloor(BANDS, enteredHowFromBelow, 5840 - VH, VH)).toBe(1);
  });

  it('is 0 for a jump (time-based easing only)', () => {
    expect(scrollFloor(BANDS, { index: 1, entry: 'jump' }, 2600, VH)).toBe(0);
  });
});

describe('stepMix', () => {
  it('fades slowly: one 60 Hz frame covers 5% of the way, so a full switch takes about a second', () => {
    expect(TONE.lerp).toBe(0.05);
    expect(stepMix(0, 1, FRAME, 0, false)).toBeCloseTo(0.05, 6);
  });

  it('never lags the scroll floor, in either direction', () => {
    expect(stepMix(0, 1, FRAME, 0.7, false)).toBeGreaterThanOrEqual(0.7);
    // 1 - 0.7 is 0.30000000000000004 in floating point, hence the epsilon.
    expect(stepMix(1, 0, FRAME, 0.7, false)).toBeLessThanOrEqual(0.3 + 1e-9);
    expect(stepMix(0, 1, FRAME, 1, false)).toBe(1);
  });

  it('snaps to the target within two seconds of frames', () => {
    let mix = 0;
    let frames = 0;
    while (mix !== 1 && frames < 200) {
      mix = stepMix(mix, 1, FRAME, 0, false);
      frames += 1;
    }
    expect(mix).toBe(1);
    expect(frames).toBeLessThanOrEqual(120);
  });

  it('is instant under reduced motion', () => {
    expect(stepMix(0.3, 0, FRAME, 0, true)).toBe(0);
  });
});
