'use client';

import { useEffect } from 'react';
import { INTRO_SEEN_KEY } from './introScript';

/**
 * Opening curtain: a small window in a dark screen opens onto the hero.
 * Pure CSS so it is on screen from the first paint; plays once per browser
 * session. The root layout marks <html data-intro="skip"> when it has already
 * played or the visitor prefers reduced motion.
 */
export function Intro() {
  useEffect(() => {
    const done = window.setTimeout(() => {
      try {
        sessionStorage.setItem(INTRO_SEEN_KEY, '1');
      } catch {
        // Private mode: the intro simply plays again next visit.
      }
      document.documentElement.dataset.intro = 'skip';
    }, 2300);
    return () => window.clearTimeout(done);
  }, []);

  return (
    <div className="ss-intro fixed inset-0 z-[100] flex items-center justify-center" aria-hidden="true">
      <div className="ss-intro-window" />
      <p className="ss-intro-word absolute left-[9vw] text-[15px] font-medium leading-tight text-white sm:text-[17px]">
        compresskit
      </p>
      <p className="ss-intro-word absolute right-[9vw] text-right text-[15px] font-medium leading-tight text-white sm:text-[17px]">
        Prune, then
        <br />
        measure.
      </p>
    </div>
  );
}
