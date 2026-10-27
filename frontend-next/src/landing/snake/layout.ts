import { TONE, type LandingLayout } from '../tone/tone';
import { buildLut, crossSegments, railSegments, toPathD, type Cubic, type Lut } from './geometry';

/**
 * Snake route, derived from the measured tone bands.
 *
 * Desktop: the line descends an outer gutter through each band (left, right,
 * left, right), crossing the page in the gap at each boundary — through a curl
 * before the results and before the closing band. The pipeline story is
 * measured on its non-sticky track, so the rail runs the full track height
 * behind the pinned panel. Mobile: one gentle rail down the right padding,
 * no crossings, no curls. Inner pages use buildRail: one calm rail down the
 * right gutter from the first band to the last, no crossings.
 */
export const SNAKE = {
  /** The head's fixed position as a fraction of the viewport height: the tone switch line. */
  headLine: TONE.switchLine,
  bandIds: ['pipeline', 'models', 'limits', 'closing'],
  loopBefore: ['models', 'closing'],
  startInset: 24,
  endInset: 48,
  crossHalf: 96,
  loopRadius: 44,
  halfWave: 360,
  maxAmplitude: 22,
  mobileBreakpoint: 768,
  mobileInset: 10,
  mobileAmplitude: 4,
  mobileHalfWave: 280,
} as const;

export interface SnakeGeometry {
  d: string;
  lut: Lut;
  width: number;
  height: number;
  mobile: boolean;
  segments: readonly Cubic[];
}

/** Centre of the outer gutter: half of (margin beside the 1320px column + its padding). */
export function gutterCenter(vw: number): number {
  const pad = vw >= 640 ? 32 : 20;
  const margin = Math.max(0, (vw - 1320) / 2);
  return Math.max(10, (margin + pad) / 2);
}

export function buildSnake(layout: LandingLayout): SnakeGeometry | null {
  const bands = SNAKE.bandIds
    .map((id) => layout.bands.find((b) => b.id === id))
    .filter((b): b is NonNullable<typeof b> => b !== undefined);
  if (bands.length < 2 || layout.vw <= 0) return null;

  const first = bands[0];
  const last = bands[bands.length - 1];
  // At max scroll the head line is at height − (1 − headLine)·vh; end above it.
  const reachable = layout.height - layout.vh * (1 - SNAKE.headLine) - 4;
  const startY = first.top + SNAKE.startInset;
  const endY = Math.min(last.bottom - SNAKE.endInset, reachable);
  if (endY <= startY) return null;

  const mobile = layout.vw < SNAKE.mobileBreakpoint;
  let segments: Cubic[];

  if (mobile) {
    segments = railSegments(layout.vw - SNAKE.mobileInset, startY, endY, SNAKE.mobileAmplitude, SNAKE.mobileHalfWave);
  } else {
    const g = gutterCenter(layout.vw);
    const amplitude = Math.min(SNAKE.maxAmplitude, Math.max(4, g * 0.45));
    const xFor = (i: number) => (i % 2 === 0 ? g : layout.vw - g);
    const loops: readonly string[] = SNAKE.loopBefore;
    segments = [];
    let y = startY;
    for (let i = 0; i < bands.length; i += 1) {
      const band = bands[i];
      const railEnd = i === bands.length - 1 ? endY : Math.max(y, band.bottom - SNAKE.crossHalf);
      segments.push(...railSegments(xFor(i), y, railEnd, amplitude, SNAKE.halfWave));
      y = Math.max(y, railEnd);
      if (i < bands.length - 1) {
        const next = bands[i + 1];
        const arrive = Math.max(y + 2 * SNAKE.crossHalf, next.top + SNAKE.crossHalf);
        segments.push(
          ...crossSegments(
            { x: xFor(i), y },
            { x: xFor(i + 1), y: arrive },
            Math.max(next.top, y + 16),
            loops.includes(next.id) ? SNAKE.loopRadius : 0,
          ),
        );
        y = arrive;
      }
    }
  }

  return { d: toPathD(segments), lut: buildLut(segments), width: layout.vw, height: layout.height, mobile, segments };
}

/**
 * Inner pages: a single rail down the right gutter from the top of the first
 * band to the bottom of the last band before the footer. Same head, same
 * follow easing as the landing snake, so every page scrolls with one gesture.
 */
export function buildRail(layout: LandingLayout): SnakeGeometry | null {
  const bands = layout.bands.filter((b) => b.id !== 'site-footer');
  if (bands.length < 2 || layout.vw <= 0) return null;
  const reachable = layout.height - layout.vh * (1 - SNAKE.headLine) - 4;
  const startY = bands[0].top + SNAKE.startInset + 96;
  const endY = Math.min(bands[bands.length - 1].bottom - SNAKE.endInset, reachable);
  if (endY <= startY) return null;
  const mobile = layout.vw < SNAKE.mobileBreakpoint;
  const x = mobile ? layout.vw - SNAKE.mobileInset : layout.vw - gutterCenter(layout.vw);
  const amplitude = mobile ? SNAKE.mobileAmplitude : Math.min(SNAKE.maxAmplitude, Math.max(4, gutterCenter(layout.vw) * 0.35));
  const segments = railSegments(x, startY, endY, amplitude, mobile ? SNAKE.mobileHalfWave : SNAKE.halfWave);
  return { d: toPathD(segments), lut: buildLut(segments), width: layout.vw, height: layout.height, mobile, segments };
}
