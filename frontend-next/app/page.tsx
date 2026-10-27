import { manrope } from '@/src/landing/fonts';
import { Closing } from '@/src/landing/Closing';
import { Hero } from '@/src/landing/Hero';
import { Intro } from '@/src/landing/Intro';
import { Limits } from '@/src/landing/Limits';
import { ModelsCompare } from '@/src/landing/ModelsCompare';
import { MotionRoot, SmoothScroll } from '@/src/landing/Motion';
import { PipelineStory } from '@/src/landing/PipelineStory';
import { SiteFooter } from '@/src/landing/SiteFooter';
import { SiteNav } from '@/src/landing/SiteNav';
import { Snake } from '@/src/landing/snake/Snake';
import { ToneRoot } from '@/src/landing/tone/ToneRoot';
import '@/src/landing/landing.css';
import '@/src/landing/tone/tone.css';

/**
 * compresskit landing page.
 * Route: `/`
 * Bands, in order: hero (light), pipeline (dark), models (light), limits (dark), closing (light).
 */
export default function HomePage() {
  return (
    <ToneRoot className={`${manrope.variable} landing relative w-full overflow-x-clip`}>
      <MotionRoot>
        <SmoothScroll />
        <Intro />
        <SiteNav />
        <Snake />
        <main>
          <Hero />
          <PipelineStory />
          <ModelsCompare />
          <Limits />
          <Closing />
        </main>
        <SiteFooter />
      </MotionRoot>
    </ToneRoot>
  );
}
