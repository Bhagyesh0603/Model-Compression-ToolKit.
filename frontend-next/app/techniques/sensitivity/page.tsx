import type { Metadata } from 'next';
import { sensitivity as s } from '@/src/data/results';
import { ProtectK } from '@/src/explore/ProtectK';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'Sensitivity — compresskit' };

export default function SensitivityPage() {
  const worst = s.layers[0];
  const best = s.layers[s.layers.length - 1];
  return (
    <TechniqueShell
      slug="sensitivity"
      title="Some layers lose much more to quantization than others"
      lede={
        <p>
          Treating every layer alike wastes the layers that carry little signal and damages the ones that carry a lot.
          The sensitivity pass measures which is which, one layer at a time, before anything is pruned or quantized.
        </p>
      }
    >
      <Section
        title="One layer at a time"
        lede={
          <p>
            Each convolution and linear layer is fake-quantized to {s.bits} bits while every other layer stays in float.
            The change in the network&apos;s output logits is measured on {s.batches} calibration batches from the
            training split, as a signal-to-quantization-noise ratio in decibels. Lower means the layer does more damage.
            It needs a handful of forward passes and no training.
          </p>
        }
      >
        <Figure
          caption={`${s.model}, ${s.layers.length} layers, worst first. ${worst.name} is the most sensitive at ${worst.sqnr_db.toFixed(1)} dB and ${best.name} the least at ${best.sqnr_db.toFixed(1)} dB.`}
        >
          <ProtectK />
          <Provenance p={s.provenance as never} />
        </Figure>
      </Section>

      <Section tone="dark"
        title="What the ranking is used for"
        lede={
          <>
            <p>
              Two things, both automatic in the budget search. Mixed precision keeps the most sensitive layers in FP32
              when quantization alone misses on accuracy. And pruning ratios are scaled down for those same layers, so
              a fragile early layer is not cut as hard as a wide redundant one. The classifier is skipped in both,
              since it is never pruned.
            </p>
          </>
        }
      />
    </TechniqueShell>
  );
}
