'use client';

import { createContext, useContext, useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { useLenis } from 'lenis/react';
import { usePrefersReducedMotion } from '../hooks';
import { LandingController, type ToneSnapshot } from './controller';

const LandingContext = createContext<LandingController | null>(null);
const SERVER_SNAPSHOT: ToneSnapshot = { tone: 'light', bandId: null };

/**
 * The .landing root. Owns the LandingController (the page's only scroll loop)
 * and exposes it to the nav and the snake. Renders no per-frame React state:
 * colours follow the --tone-dark CSS number the controller writes.
 */
export function ToneRoot({ className, children }: { className?: string; children: ReactNode }) {
  const [controller] = useState(() => new LandingController());
  const ref = useRef<HTMLDivElement>(null);
  const reduce = usePrefersReducedMotion();
  const lenis = useLenis();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    controller.attach(el);
    return () => controller.detach();
  }, [controller]);

  useEffect(() => {
    controller.setLenis(lenis ?? null);
  }, [controller, lenis]);

  useEffect(() => {
    controller.setReducedMotion(reduce);
  }, [controller, reduce]);

  return (
    <LandingContext.Provider value={controller}>
      <div ref={ref} className={className} data-tone="light" data-tone-settled="true">
        {children}
      </div>
    </LandingContext.Provider>
  );
}

export function useLandingController(): LandingController {
  const controller = useContext(LandingContext);
  if (!controller) throw new Error('useLandingController must be used inside <ToneRoot>');
  return controller;
}

/** The active band and its tone. Changes only at band switches, never per frame. */
export function useLandingTone(): ToneSnapshot {
  const controller = useLandingController();
  return useSyncExternalStore(controller.subscribeTone, controller.getToneSnapshot, () => SERVER_SNAPSHOT);
}
