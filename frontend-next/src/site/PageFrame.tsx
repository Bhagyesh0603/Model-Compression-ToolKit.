'use client';

import type { ReactNode } from 'react';
import { manrope } from '@/src/landing/fonts';
import { MotionRoot, SmoothScroll } from '@/src/landing/Motion';
import { Snake } from '@/src/landing/snake/Snake';
import { SiteFooter } from '@/src/landing/SiteFooter';
import { SiteNav } from '@/src/landing/SiteNav';
import { ToneRoot } from '@/src/landing/tone/ToneRoot';
import '@/src/landing/landing.css';
import '@/src/landing/tone/tone.css';

/** Nav, footer and the landing token root around an inner page (one light band, no intro). */
export function PageFrame({ children }: { children: ReactNode }) {
  return (
    <ToneRoot className={`${manrope.variable} landing relative w-full overflow-x-clip`}>
      <MotionRoot>
        <SmoothScroll />
        <SiteNav />
        <Snake variant="rail" />
        <main className="pt-[72px]">{children}</main>
        <SiteFooter compact />
      </MotionRoot>
    </ToneRoot>
  );
}
