import type { Metadata } from 'next';
import { bnRecal as bn, pct } from '@/src/data/results';
import { BnShift } from '@/src/site/charts/BnShift';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'BatchNorm recalibration — compresskit' };

export default function RecalibrationPage() {
  const a = bn.accuracy;
  return (
    <TechniqueShell
      slug="recalibration"
      title="Removing channels leaves BatchNorm statistics stale"
      lede={
        <p>
          A BatchNorm layer normalizes with a running mean and variance collected while the full network was trained.
          Once channels are gone, the activations feeding it change, and those stored statistics no longer describe
          them.
        </p>
      }
    >
      <Section
        title="Re-estimate, don't train"
        lede={
          <p>
            Recalibration resets every BatchNorm layer&apos;s running statistics, switches the momentum to a cumulative
            average so the result is the exact mean over everything it sees, and runs forward passes over calibration
            data. There is no backpropagation and no weight changes. The data comes from the training split only, because
            these statistics are fitted to whatever they are shown.
          </p>
        }
      >
        <Figure
          caption={`One layer (${bn.layer}) of ResNet-20 after a one-shot ${Math.round(bn.prune_ratio * 100)}% prune: ${bn.channels_before} channels became ${bn.channels_after}. Each pair of dots is one surviving channel's running variance, before and after re-estimation.`}
        >
          <BnShift
            indices={bn.kept_channel_indices}
            original={bn.running_var.original}
            recalibrated={bn.running_var.recalibrated}
            className="w-full max-w-[640px]"
          />
          <Provenance p={bn.provenance as never} />
        </Figure>
      </Section>

      <Section tone="dark"
        title="What it buys, honestly"
        lede={
          <p>
            Recalibration is cheap and always worth doing before measuring a pruned network, but on a cut this large it
            does not restore the model. Fine-tuning does the real recovery, which is why the pipeline does both.
          </p>
        }
      >
        <Figure caption="Top-1 on the CIFAR-10 test split. Same pruned network in the two middle states; only the BatchNorm statistics differ.">
          <dl className="figures-tab grid gap-6 sm:grid-cols-3">
            {[
              ['Before pruning', a.baseline],
              ['Right after the cut, stale statistics', a.after_cut_stale_bn],
              ['After re-estimating BatchNorm', a.after_bn_recalibration],
            ].map(([label, v]) => (
              <div key={label as string}>
                <dt className="text-[13px] text-(--muted)">{label}</dt>
                <dd className="mt-1 text-[clamp(30px,3vw,44px)] leading-none tracking-[-0.04em]">{pct(v as number)}</dd>
              </div>
            ))}
          </dl>
          <Provenance p={bn.provenance as never} />
        </Figure>
      </Section>
    </TechniqueShell>
  );
}
