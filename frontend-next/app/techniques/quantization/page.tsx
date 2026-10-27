import type { Metadata } from 'next';
import { hosts, models, quantWeights as q, finalStage, ms, mb, pct } from '@/src/data/results';
import { BitsExplorer } from '@/src/explore/BitsExplorer';
import { RoundingScatter } from '@/src/site/charts/RoundingScatter';
import { ScaleBars } from '@/src/site/charts/ScaleBars';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'Quantization — compresskit' };

export default function QuantizationPage() {
  const withInt8 = models.filter((m) => finalStage(m).precision === 'int8' && !m.schedule.includes('short smoke run'));
  const maxScale = Math.max(...q.scales_per_output_channel);
  const minScale = Math.min(...q.scales_per_output_channel);

  return (
    <TechniqueShell
      slug="quantization"
      title="Quantization runs last, and only where it is fast"
      lede={
        <p>
          Quantization stores weights as 8-bit integers. That shrinks the file about fourfold, and on a CPU with fused
          integer dot-product instructions it also runs faster. It changes the weight type, so it is the one step that
          cannot be followed by pruning or by gradient training.
        </p>
      }
    >
      <Section
        title="What rounding does to one channel"
        lede={
          <p>
            Weights are quantized per output channel: each channel gets its own scale, so a channel of small weights is
            not forced onto the grid of the largest filter in the layer. Every weight then snaps to the nearest of 255
            steps, which bounds its error to half a step.
          </p>
        }
      >
        <div className="mb-6">
          <Figure caption="Drag the bit width down. Each line is one weight moving from its FP32 value to the nearest level of the integer grid; fewer bits means fewer levels, wider steps and larger error.">
            <BitsExplorer />
            <Provenance p={q.provenance as never} />
          </Figure>
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <Figure
            caption={`Channel 0 of ${q.layer} (${q.model}): ${q.channel0.weights.length} weights, scale ${q.channel0.scale.toExponential(2)}. The error is in units of one step, so every point stays within plus or minus 0.5.`}
          >
            <RoundingScatter weights={q.channel0.weights} scale={q.channel0.scale} className="w-full" />
            <Provenance p={q.provenance as never} />
          </Figure>
          <Figure
            caption={`The ${q.scales_per_output_channel.length} per-channel scales of the same layer range from ${minScale.toExponential(2)} to ${maxScale.toExponential(2)}. The whole layer used ${q.distinct_int_levels_used} of the 256 integer codes, and the mean squared weight error is ${q.weight_mse.toExponential(2)}.`}
          >
            <ScaleBars scales={q.scales_per_output_channel} className="w-full" />
            <Provenance p={q.provenance as never} />
          </Figure>
        </div>
      </Section>

      <Section tone="dark"
        title="INT8 is only built where it pays"
        lede={
          <p>
            The toolkit probes the CPU for fused INT8 instructions (VNNI, AMX, dot-product, i8mm). Without them INT8 can
            be slower than FP32, so it falls back to FP16 and refuses to build INT8 unless you pass an override. On the
            CPU provider FP16 is cast back to float32 for the math: it halves the file but is not a speed win.
          </p>
        }
      >
        <Figure caption="The two machines the figures on this site were measured on. A speedup on one says nothing about the other.">
          <div className="grid gap-4 md:grid-cols-2">
            {hosts.map((h) => (
              <div key={h.label} className="rounded-2xl bg-(--chip) p-5">
                <p className="text-[12px] text-(--muted)">{h.label}</p>
                <p className="mt-1 text-[19px] tracking-[-0.02em]">{h.cpu}</p>
                <p className="mt-3 flex flex-wrap gap-1.5 font-mono text-[11px]">
                  {h.features.map((f) => (
                    <span key={f} className="rounded-md border border-(--line-strong) px-1.5 py-0.5">
                      {f}
                    </span>
                  ))}
                </p>
                <p className="mt-4 text-[14px]">
                  INT8 kernels: <span className="font-semibold">{h.int8_speed}</span>
                </p>
                {'int8_note' in h ? <p className="mt-1 text-[12px] text-(--muted)">{(h as { int8_note: string }).int8_note}</p> : null}
                <Provenance p={h.provenance as never} />
              </div>
            ))}
          </div>
        </Figure>
      </Section>

      <Section
        title="What it did to each model"
        lede={
          <p>
            Each row compares the pruned FP32 network with the same network after INT8 quantization, both measured on
            the exported file. The accuracy cost is on the same line as the size and latency change.
          </p>
        }
      >
        <Figure caption="Pruned FP32 (the stage just before quantization) to INT8. The Kaggle CPU has no fast INT8 kernels, so its INT8 latency is not what an INT8-capable CPU would show.">
          <table className="figures-tab w-full text-left text-[14px]">
            <thead className="text-[12px] text-(--muted)">
              <tr>
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">Top-1</th>
                <th className="pb-2 text-right font-medium">Size</th>
                <th className="pb-2 text-right font-medium">Median latency</th>
                <th className="pb-2 pl-4 font-medium">Measured on</th>
              </tr>
            </thead>
            <tbody>
              {withInt8.map((m) => {
                const before = m.stages[m.stages.length - 2];
                const after = finalStage(m);
                return (
                  <tr key={m.arch} className="border-t border-(--line)">
                    <td className="py-3">{m.name}</td>
                    <td className="py-3 text-right">
                      {pct(before.accuracy)} to {pct(after.accuracy)}
                    </td>
                    <td className="py-3 text-right">
                      {mb(before.size_mb)} to {mb(after.size_mb)}
                    </td>
                    <td className="py-3 text-right">
                      {ms(before.median_ms)} to {ms(after.median_ms)}
                    </td>
                    <td className="py-3 pl-4 text-[12px] text-(--muted)">
                      {m.provenance.host} ({m.provenance.kind})
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Figure>
      </Section>

      <Section
        title="When post-training INT8 loses too much"
        lede={
          <p>
            Quantization-aware training runs a short fine-tune with the weights fake-quantized in the forward pass. Rounding
            has zero gradient, so the backward pass uses the straight-through estimator: the forward value is the
            rounded one, the gradient flows as if rounding were the identity. The budget search tries it only after
            cheaper options miss on accuracy.
          </p>
        }
      />
    </TechniqueShell>
  );
}
