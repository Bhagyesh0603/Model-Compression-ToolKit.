'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, useMotionValue, useTransform } from 'motion/react';
import { usePrefersReducedMotion } from '../hooks';
import { useLandingController } from '../tone/ToneRoot';
import { lengthAtDepth, pointAtLength, stepFollow, type Follow } from './geometry';
import { SNAKE, buildRail, buildSnake, type SnakeGeometry } from './layout';

/**
 * The scroll-drawn line. Geometry is rebuilt only when the controller
 * re-measures; each frame does two binary searches in the lookup table and
 * sets three MotionValues — no DOM reads, no React renders.
 */
export function Snake({ variant = 'landing' }: { variant?: 'landing' | 'rail' }) {
  const controller = useLandingController();
  const reduce = usePrefersReducedMotion();
  const [geom, setGeom] = useState<SnakeGeometry | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const progress = useMotionValue(0);
  const headX = useMotionValue(0);
  const headY = useMotionValue(0);
  const headOpacity = useMotionValue(0);
  // A zero-length dash with round caps still paints a dot; hide it until there is a line.
  const drawOpacity = useTransform(progress, (p) => (p > 0.0005 ? 0.9 : 0));

  useEffect(
    () => controller.onLayout((layout) => setGeom(variant === 'rail' ? buildRail(layout) : buildSnake(layout))),
    [controller, variant],
  );

  useEffect(() => {
    if (!geom || reduce) return;
    const { lut } = geom;
    let follow: Follow = { drawn: progress.get() * lut.total, catching: false };
    let settled: boolean | null = null;
    return controller.onFrame((frame) => {
      const target = lengthAtDepth(lut, frame.top + frame.vh * SNAKE.headLine);
      follow = stepFollow(follow, target, frame.dt);
      const p = pointAtLength(lut, follow.drawn);
      progress.set(lut.total > 0 ? follow.drawn / lut.total : 0);
      headX.set(p.x);
      headY.set(p.y);
      headOpacity.set(follow.drawn > 1 && follow.drawn < lut.total - 1 ? 1 : 0);
      const nowSettled = !follow.catching;
      if (nowSettled !== settled && svgRef.current) {
        svgRef.current.dataset.snakeSettled = String(nowSettled);
        settled = nowSettled;
      }
      return follow.catching;
    });
  }, [controller, geom, reduce, progress, headX, headY, headOpacity]);

  if (!geom) return null;

  return (
    <svg
      ref={svgRef}
      data-snake
      data-snake-settled="false"
      aria-hidden="true"
      focusable="false"
      className="ss-snake pointer-events-none absolute left-0 top-0 z-0"
      width={geom.width}
      height={geom.height}
      viewBox={`0 0 ${geom.width} ${geom.height}`}
    >
      <path
        data-snake-guide
        d={geom.d}
        fill="none"
        className="stroke-(--snake)"
        strokeOpacity={reduce ? 0.28 : 0.12}
        strokeWidth={geom.mobile ? 1 : 1.5}
      />
      {!reduce && (
        <>
          <motion.path
            data-snake-draw
            d={geom.d}
            fill="none"
            className="stroke-(--snake)"
            strokeWidth={geom.mobile ? 2 : 3}
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ pathLength: progress, opacity: drawOpacity }}
          />
          <motion.g data-snake-head style={{ x: headX, y: headY, opacity: headOpacity }}>
            <circle r={geom.mobile ? 6 : 10} className="fill-(--snake)" fillOpacity={0.22} />
            <circle r={geom.mobile ? 3 : 4.5} className="fill-(--snake)" />
          </motion.g>
        </>
      )}
    </svg>
  );
}
