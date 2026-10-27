import { PageFrame } from '@/src/site/PageFrame';
import { Band, Inner } from '@/src/motion/Band';
import { BudgetCheck } from '@/src/explore/BudgetCheck';
import { StageExplorer } from '@/src/explore/StageExplorer';

export const metadata = { title: 'Explore the results | compresskit' };

export default function ExplorePage() {
  return (
    <PageFrame>
      <Band tone="light" labelledBy="explore-title">
        <Inner className="!pb-14 !pt-[116px] sm:!pb-16 sm:!pt-[140px]">
          <p className="label-xs">Explore</p>
          <h1 id="explore-title" className="mt-4 max-w-[16ch] text-balance text-[clamp(36px,4.8vw,68px)] font-normal leading-[1] tracking-[-0.05em]">
            Pick a model, scrub the stages.
          </h1>
          <p className="mt-6 max-w-[58ch] text-[17px] leading-[1.65] text-(--muted)">
            Every figure here is a recorded measurement of the exported ONNX file. Nothing is interpolated: the controls
            move between stages that were actually run.
          </p>
        </Inner>
      </Band>
      <Band id="explorer" tone="dark">
        <Inner>
          <StageExplorer />
        </Inner>
      </Band>
      <Band id="budget" tone="light" labelledBy="budget-title">
        <Inner>
          <p className="label-xs">Budget check</p>
          <h2 id="budget-title" className="mt-4 max-w-[20ch] text-balance text-[clamp(30px,3.8vw,54px)] font-normal leading-[1.05] tracking-[-0.045em]">
            Would any recorded run meet your limits?
          </h2>
          <div className="mt-12">
            <BudgetCheck />
          </div>
        </Inner>
      </Band>
    </PageFrame>
  );
}
