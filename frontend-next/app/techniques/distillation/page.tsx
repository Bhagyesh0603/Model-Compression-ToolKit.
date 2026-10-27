import type { Metadata } from 'next';
import { distillation as d } from '@/src/data/results';
import { TempExplorer } from '@/src/explore/TempExplorer';
import { Provenance } from '@/src/site/Provenance';
import { Figure, Section } from '@/src/site/Section';
import { TechniqueShell } from '@/src/site/TechniqueShell';

export const metadata: Metadata = { title: 'Distillation — compresskit' };

export default function DistillationPage() {
  const total = d.dkd_total;
  const tPart = d.alpha * d.tckd;
  const nPart = d.beta * d.nckd;
  return (
    <TechniqueShell
      slug="distillation"
      title="The original model teaches the pruned one"
      lede={
        <p>
          Fine-tuning a pruned network on hard labels alone throws away what the original knew about which wrong
          answers are almost right. Distillation keeps that: the untouched original is the teacher, and the pruned
          network is trained to match its output distribution.
        </p>
      }
    >
      <Section
        title="One image, both models"
        lede={
          <p>
            The student here is the one-shot pruned and BatchNorm-recalibrated ResNet-20 before any fine-tuning. This is
            a real CIFAR-10 test image the teacher classifies correctly and the student gets wrong. Raising the
            temperature flattens both distributions, which exposes the ranking among the non-target classes.
          </p>
        }
      >
        <Figure
          caption={`True class: ${d.classes[d.target]}. Slide the temperature: at T = 1 the teacher is nearly certain and the other classes are invisible; at higher T the ranking among them shows, and that ranking is most of what the student can learn.`}
        >
          <TempExplorer />
          <Provenance p={d.provenance as never} />
        </Figure>
      </Section>

      <Section tone="dark"
        title="Plain KD and decoupled KD"
        lede={
          <>
            <p>
              Plain KD is the KL divergence between temperature-softened teacher and student, scaled by T squared so the
              gradient size does not shrink as T grows. Decoupled KD splits it in two: one term about the target class
              against everything else (TCKD), and one about how the remaining classes rank among themselves (NCKD, with
              the target masked out before the softmax). They are weighted separately, <span className="font-mono text-[14px]">alpha·TCKD + beta·NCKD</span>.
            </p>
          </>
        }
      >
        <Figure caption={`The same image and models. Defaults in the toolkit: T = ${d.temperature}, alpha = ${d.alpha}, beta = ${d.beta}. In this example the non-target term carries most of the loss.`}>
          <table className="figures-tab w-full text-left text-[14px]">
            <thead className="text-[12px] text-(--muted)">
              <tr>
                <th className="pb-2 font-medium">Term</th>
                <th className="pb-2 text-right font-medium">Raw value</th>
                <th className="pb-2 text-right font-medium">Weight</th>
                <th className="pb-2 text-right font-medium">Contribution</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-(--line)">
                <td className="py-3">TCKD (target vs the rest)</td>
                <td className="py-3 text-right">{d.tckd.toFixed(3)}</td>
                <td className="py-3 text-right">{d.alpha}</td>
                <td className="py-3 text-right">{tPart.toFixed(3)}</td>
              </tr>
              <tr className="border-t border-(--line)">
                <td className="py-3">NCKD (ranking of non-target classes)</td>
                <td className="py-3 text-right">{d.nckd.toFixed(3)}</td>
                <td className="py-3 text-right">{d.beta}</td>
                <td className="py-3 text-right">{nPart.toFixed(3)}</td>
              </tr>
              <tr className="border-t border-(--line) font-medium">
                <td className="py-3">Decoupled KD total</td>
                <td className="py-3" />
                <td className="py-3" />
                <td className="py-3 text-right">{total.toFixed(3)}</td>
              </tr>
              <tr className="border-t border-(--line)">
                <td className="py-3 text-(--muted)">For comparison: plain KD (T² · KL)</td>
                <td className="py-3" />
                <td className="py-3" />
                <td className="py-3 text-right text-(--muted)">{d.kd_total.toFixed(3)}</td>
              </tr>
            </tbody>
          </table>
          <Provenance p={d.provenance as never} />
        </Figure>
      </Section>
    </TechniqueShell>
  );
}
