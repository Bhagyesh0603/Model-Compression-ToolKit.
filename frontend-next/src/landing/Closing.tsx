import Link from 'next/link';
import { ArrowUpRight, BookOpen, Cpu, Terminal } from 'lucide-react';
import { Reveal } from './Motion';

const POINTS = [
  { icon: Terminal, text: 'One command, or one Kaggle notebook with Run All' },
  { icon: Cpu, text: 'Check the latency on your own CPU with compresskit bench' },
  { icon: BookOpen, text: 'Tested on ResNet and VGG-style classifiers' },
] as const;

/** The last light band: a low-pressure invitation before the dark footer. */
export function Closing() {
  return (
    <section
      id="closing"
      data-tone-band
      data-tone="light"
      aria-labelledby="closing-title"
      className="relative z-10 scroll-mt-[72px]"
    >
      <div className="mx-auto max-w-[1320px] px-5 pb-28 pt-28 sm:px-8 sm:pb-36 sm:pt-36">
        <Reveal>
          <h2
            id="closing-title"
            className="max-w-[16ch] text-balance text-[clamp(36px,4.8vw,66px)] leading-[1.03] tracking-[-0.045em]"
          >
            Run it on your model, then check the file yourself.
          </h2>
        </Reveal>

        <div className="mt-8 grid gap-10 lg:grid-cols-[1.4fr_1fr] lg:items-end">
          <Reveal delay={0.08}>
            <p className="max-w-[520px] text-[16px] leading-[1.65] text-(--muted)">
              Clone the repository, point it at a trained image-classification CNN, and get back an ONNX file and a
              report. MobileNet-style models are untested and detection or segmentation models are unsupported.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/run"
                className="group inline-flex h-[52px] items-center gap-4 rounded-full bg-(--inv-bg) pl-6 pr-1.5 text-[15px] font-medium text-(--inv-fg) transition-colors hover:bg-(--inv-hover)"
              >
                How to run it
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-600 text-white transition-transform duration-300 group-hover:rotate-45">
                  <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                </span>
              </Link>
              <Link
                href="/explore"
                className="inline-flex h-[52px] items-center gap-2.5 rounded-full border border-(--line-strong) px-6 text-[15px] font-medium transition-colors hover:bg-(--hover)"
              >
                Explore the results
              </Link>
              <Link
                href="/notebooks"
                className="inline-flex h-[52px] items-center gap-2.5 rounded-full border border-(--line-strong) px-6 text-[15px] font-medium transition-colors hover:bg-(--hover)"
              >
                <BookOpen className="h-4 w-4 text-(--muted)" aria-hidden="true" />
                The notebooks
              </Link>
            </div>
          </Reveal>

          <Reveal delay={0.16}>
            <ul className="divide-y divide-(--line) border-y border-(--line)">
              {POINTS.map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-center gap-3 py-4 text-[15px]">
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-(--chip) text-brand-600">
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </span>
                  <span>{text}</span>
                </li>
              ))}
            </ul>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
