import { bnRecal, gradualVsOneShot as gv, models, pct, search } from '@/src/data/results';
import { Band, Inner } from '@/src/motion/Band';
import { AnimBar } from '@/src/viz/AnimBar';
import { Reveal } from './Motion';

const R20 = models.find((m) => m.arch === 'resnet20')!;
const unmet = search.runs.find((r) => !r.met);

/** What did not go the way it was expected to, with the numbers. */
export function Limits() {
  const gradual = gv.gradual.history[gv.gradual.history.length - 1].acc_after_finetune;
  const oneShot = gv.one_shot.history[gv.one_shot.history.length - 1].acc_after_finetune;
  return (
    <Band id="limits" tone="dark" labelledBy="limits-title">
      <Inner>
        <Reveal>
          <p className="label-xs !text-(--muted)">What did not go as expected</p>
          <h2 id="limits-title" className="mt-4 max-w-[20ch] text-balance text-[clamp(32px,4.2vw,60px)] font-normal leading-[1.04] tracking-[-0.045em]">
            The results that did not favour the method stay in.
          </h2>
        </Reveal>
        <div className="mt-12 grid gap-5 lg:grid-cols-3">
          <Reveal>
            <Item
              title="Gradual pruning did not beat one-shot"
              body={`With the same ${gv.total_fine_tune_epochs} fine-tuning epochs on ResNet-20, one cut then fine-tune ended higher. One run each, so the gap is not established.`}
            >
              <AnimBar label={`Gradual, ${gv.gradual.steps} steps`} value={gradual} display={pct(gradual)} />
              <AnimBar label="One-shot" value={oneShot} display={pct(oneShot)} accent delay={0.1} />
            </Item>
          </Reveal>
          <Reveal delay={0.08}>
            <Item
              title="BatchNorm re-estimation helps a little"
              body="After a 50% cut with no fine-tuning, accuracy is near chance. Re-estimating the statistics lifts it, but fine-tuning is still needed."
            >
              <AnimBar label="Original" value={bnRecal.accuracy.baseline} display={pct(bnRecal.accuracy.baseline)} />
              <AnimBar label="Cut, stale BatchNorm" value={bnRecal.accuracy.after_cut_stale_bn} display={pct(bnRecal.accuracy.after_cut_stale_bn)} delay={0.1} />
              <AnimBar label="Cut, re-estimated" value={bnRecal.accuracy.after_bn_recalibration} display={pct(bnRecal.accuracy.after_bn_recalibration)} accent delay={0.2} />
            </Item>
          </Reveal>
          <Reveal delay={0.16}>
            <Item
              title="Compression costs accuracy"
              body={`ResNet-20 lost ${R20.totals.accuracy_drop_pp.toFixed(2)} points. A budget can also be unmet: asked for ${unmet?.described ?? 'an impossible budget'}, the search says so and returns its best trade-off.`}
            >
              <AnimBar label="ResNet-20 before" value={R20.baseline.accuracy} display={pct(R20.baseline.accuracy)} />
              <AnimBar label="ResNet-20 after INT8" value={R20.stages[2].accuracy} display={pct(R20.stages[2].accuracy)} accent delay={0.1} />
            </Item>
          </Reveal>
        </div>
      </Inner>
    </Band>
  );
}

function Item({ title, body, children }: { title: string; body: string; children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col rounded-[24px] border border-(--line) p-6">
      <h3 className="text-[22px] leading-[1.15] tracking-[-0.03em]">{title}</h3>
      <p className="mt-3 text-[14px] leading-[1.65] text-(--muted)">{body}</p>
      <div className="mt-auto space-y-4 pt-6">{children}</div>
    </div>
  );
}
