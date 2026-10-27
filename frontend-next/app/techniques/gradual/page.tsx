import type { Metadata } from 'next';
import { gradualVsOneShot as g, modelBySlug, pct } from '@/src/data/results';
import { GradualPlayer } from '@/src/explore/GradualPlayer';
import { Sawtooth } from '@/src/site/charts/Sawtooth';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'Gradual pruning — compresskit' };

export default function GradualPage() {
  const vgg = modelBySlug('vgg16-bn');
  const gradualFinal = g.gradual.history[g.gradual.history.length - 1].acc_after_finetune;
  const oneShotFinal = g.one_shot.history[0].acc_after_finetune;
  const diff = (oneShotFinal - gradualFinal) * 100;
  const worstGradualCut = Math.min(...g.gradual.history.map((h) => h.acc_after_cut));

  return (
    <TechniqueShell
      slug="gradual"
      title="Gradual pruning cuts in steps and fine-tunes after each"
      lede={
        <p>
          Instead of removing all the channels at once, the cut is split into steps. After each step BatchNorm
          statistics are re-estimated and the network is fine-tuned against the untouched original as teacher, then the
          next step starts from the recovered model.
        </p>
      }
    >
      <Section
        title="What one run looks like"
        lede={
          <p>
            Open dots are accuracy right after a cut, solid dots are after the fine-tuning that follows. Every value is
            from the run&apos;s own <span className="font-mono text-[14px]">prune_history</span>.
          </p>
        }
      >
        <div className="space-y-6">
          <Figure caption={`ResNet-20, ${g.gradual.steps} steps of ${g.gradual.epochs_per_step} DKD epochs each, 50% of channels. Baseline ${pct(g.baseline_accuracy)}.`}>
            <GradualPlayer />
            <Provenance p={g.provenance as never} />
          </Figure>
          {vgg ? (
            <Figure caption={`VGG-16-BN, ${vgg.prune_history.length} steps of 5 DKD epochs each, 50% of channels. Baseline ${pct(vgg.baseline.accuracy)}.`}>
              <Sawtooth baseline={vgg.baseline.accuracy} history={vgg.prune_history} className="mx-auto w-full max-w-[640px]" summary="VGG-16-BN accuracy through five gradual pruning steps" />
              <Provenance p={vgg.provenance} source={vgg.source} />
            </Figure>
          ) : null}
        </div>
      </Section>

      <Section tone="dark"
        title="Against one big cut"
        lede={
          <p>
            Same model, same 50% of channels, same total of {g.total_fine_tune_epochs} fine-tuning epochs with the same
            loss. The only difference is whether the cut happens in {g.gradual.steps} steps or in one.
          </p>
        }
      >
        <Figure
          caption={
            <>
              Right after the cut, one-shot sits at {pct(g.one_shot.history[0].acc_after_cut)} while each gradual step
              stays above {pct(worstGradualCut)}, so gradual steps are less destructive. But after the same fine-tuning
              budget, one-shot finished {Math.abs(diff).toFixed(2)} points {diff > 0 ? 'higher' : 'lower'} than gradual.
              Each configuration was run once, so a difference this size is within what a different random seed could
              move. This page does not claim gradual pruning is more accurate; on this model it was not.
            </>
          }
        >
          <table className="figures-tab w-full text-left text-[14px]">
            <thead className="text-[12px] text-(--muted)">
              <tr>
                <th className="pb-2 font-medium">Schedule</th>
                <th className="pb-2 text-right font-medium">Worst accuracy right after a cut</th>
                <th className="pb-2 text-right font-medium">Final accuracy</th>
                <th className="pb-2 text-right font-medium">Fine-tune epochs</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-(--line)">
                <td className="py-3">Gradual, {g.gradual.steps} steps</td>
                <td className="py-3 text-right">{pct(worstGradualCut)}</td>
                <td className="py-3 text-right">{pct(gradualFinal)}</td>
                <td className="py-3 text-right">{g.total_fine_tune_epochs}</td>
              </tr>
              <tr className="border-t border-(--line)">
                <td className="py-3">One-shot</td>
                <td className="py-3 text-right">{pct(g.one_shot.history[0].acc_after_cut)}</td>
                <td className="py-3 text-right">{pct(oneShotFinal)}</td>
                <td className="py-3 text-right">{g.total_fine_tune_epochs}</td>
              </tr>
            </tbody>
          </table>
          <Provenance p={g.provenance as never} />
        </Figure>
      </Section>
    </TechniqueShell>
  );
}
