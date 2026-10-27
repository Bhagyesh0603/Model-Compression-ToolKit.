'use client';

import { MotionConfig, motion } from 'motion/react';
import { ReactLenis } from 'lenis/react';
import 'lenis/dist/lenis.css';
import { usePrefersReducedMotion } from './hooks';
import { ANCHOR_OFFSET, LENIS_LERP } from './scroll';

const EASE = [0.2, 0.7, 0.2, 1] as const;

/** Motion defaults for the landing page; honours the OS reduced-motion setting. */
export function MotionRoot({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}

/** Root smooth scrolling. Not mounted at all for reduced-motion users. */
export function SmoothScroll() {
  const reduce = usePrefersReducedMotion();
  if (reduce) return null;
  return <ReactLenis root options={{ lerp: LENIS_LERP, anchors: { offset: ANCHOR_OFFSET } }} />;
}

type RevealProps = {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  y?: number;
};

/** Fades and lifts content in the first time it scrolls into view. */
export function Reveal({ children, className, delay = 0, y = 28 }: RevealProps) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y, filter: 'blur(8px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: '0px 0px -10% 0px' }}
      transition={{ duration: 0.9, ease: EASE, delay }}
    >
      {children}
    </motion.div>
  );
}
