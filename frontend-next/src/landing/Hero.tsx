'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { ArrowUpRight, RotateCcw } from 'lucide-react';
import { animate, useMotionValue, useTransform } from 'motion/react';
import { count, finalStage, mb, models, ms, pct } from '@/src/data/results';
import { MotionNumber } from '@/src/motion/MotionNumber';
import { Band } from '@/src/motion/Band';
import { EASE, lerp } from '@/src/motion/tokens';
import { ModelStrip } from '@/src/viz/ModelStrip';
import { layersOf } from '@/src/viz/layers';
import { usePrefersReducedMotion } from './hooks';
import { IsoScene } from './IsoScene';

const MODEL = models.find((m) => m.arch === 'resnet20')!;
const LAYERS = layersOf(MODEL)!;
const BASE = MODEL.baseline;
const FINAL = finalStage(MODEL);
const STATS = MODEL.prune_stats!;
const HEADLINE = 'Shrink a CNN, then measure what it cost.';

const fmtCount = (v: number) => count(Math.round(v));
const fmtPct = (v: number) => pct(v);
const fmtMb = (v: number) => mb(v);
const fmtMs = (v: number) => ms(v);

/**
 * First screen: ResNet-20 as columns of channels. On load the recorded 50%
 * cut plays, then the INT8 pass; the four numbers beneath move between the
 * baseline and the final recorded measurements.
 */
export function Hero() {
  const reduce = usePrefersReducedMotion();
  const cut = useMotionValue(reduce ? 1 : 0);
  const quant = useMotionValue(reduce ? 1 : 0);

  // The backdrop's floating boxes stay still until the entrance has finished, so only one thing animates at a time.
  const [calm, setCalm] = useState(true);
  useEffect(() => {
    const t = window.setTimeout(() => setCalm(false), document.documentElement.dataset.intro === 'skip' ? 2800 : 4600);
    return () => window.clearTimeout(t);
  }, []);

  const play = useCallback(
    (replay = false) => {
      cut.set(0);
      quant.set(0);
      // First play waits for the curtain, the text and the card (about 3.7s on a first visit); a replay starts at once.
      const wait = replay ? 0.1 : document.documentElement.dataset.intro === 'skip' ? 1.9 : 4.1;
      const a = animate(cut, 1, { duration: 1.9, ease: EASE, delay: wait });
      const b = animate(quant, 1, { duration: 1, ease: EASE, delay: wait + 2.2 });
      return () => {
        a.stop();
        b.stop();
      };
    },
    [cut, quant],
  );

  useEffect(() => {
    if (reduce) {
      cut.set(1);
      quant.set(1);
      return;
    }
    return play();
  }, [reduce, play, cut, quant]);

  const overall = useTransform([cut, quant], ([c, q]: number[]) => (c + q) / 2);
  const params = useTransform(cut, (c) => lerp(STATS.params_before, STATS.params_after, c));
  const size = useTransform(overall, (p) => lerp(BASE.size_mb, FINAL.size_mb, p));
  const acc = useTransform(overall, (p) => lerp(BASE.accuracy, FINAL.accuracy, p));
  const lat = useTransform(overall, (p) => lerp(BASE.median_ms, FINAL.median_ms, p));

  return (
    <Band id="hero" tone="light" labelledBy="hero-title" className="overflow-hidden">
      {/* The isometric block field is the landing page's backdrop only. A veil keeps text readable and fades it into the next band. */}
      <div aria-hidden="true" className={`pointer-events-none absolute inset-0 ${calm ? 'ss-iso-calm' : ''}`}>
        <IsoScene className="h-full w-full" />
        <div className="absolute inset-0 bg-gradient-to-r from-(--bg)/90 via-(--bg)/55 to-(--bg)/10" />
        <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-(--bg) to-transparent" />
      </div>
      <div className="relative z-10 mx-auto w-full max-w-[1320px] px-5 pb-20 pt-[132px] sm:px-8 sm:pb-28 sm:pt-[156px]">
        <p className="label-xs ss-rise">Model compression toolkit</p>
        <h1
          id="hero-title"
          aria-label={HEADLINE}
          className="mt-5 max-w-[15ch] text-balance text-[clamp(44px,6.6vw,104px)] font-normal leading-[0.98] tracking-[-0.05em]"
        >
          {HEADLINE.split(' ').map((w, i) => (
            <span key={i} aria-hidden="true" className="ss-rise inline-block" style={{ '--ss-d': `${0.12 + i * 0.075}s` } as React.CSSProperties}>
              {w}
              {i < HEADLINE.split(' ').length - 1 ? '\u00a0' : ''}
            </span>
          ))}
        </h1>
        <div className="mt-8 flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
          <p className="ss-fade max-w-[54ch] text-[17px] leading-[1.65] text-(--muted)" style={{ '--ss-d': '0.75s' } as React.CSSProperties}>
            compresskit prunes, distils and quantizes an image classifier, then measures the exported ONNX file rather
            than the model in memory. Below is ResNet-20 on CIFAR-10, drawn from the recorded run: each column is a
            layer and each cell is a channel.
          </p>
          <div className="ss-fade flex flex-wrap items-center gap-3" style={{ '--ss-d': '0.95s' } as React.CSSProperties}>
            <Link
              href="/explore"
              className="group inline-flex h-[52px] items-center gap-4 rounded-full bg-(--inv-bg) pl-6 pr-1.5 text-[15px] font-medium text-(--inv-fg) transition-colors hover:bg-(--inv-hover)"
            >
              Explore the results
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-600 text-white transition-transform duration-300 group-hover:rotate-45">
                <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
              </span>
            </Link>
            <a
              href="#pipeline"
              className="inline-flex h-[52px] items-center rounded-full border border-(--line-strong) px-6 text-[15px] font-medium transition-colors hover:bg-(--hover)"
            >
              How a run goes
            </a>
          </div>
        </div>

        <div className="ss-fade mt-14 rounded-[28px] border border-(--line) bg-(--bg)/90 p-5 sm:p-8" style={{ '--ss-d': '1.15s' } as React.CSSProperties}>
          <div className="flex flex-wrap items-center justify-between gap-3 text-[12px] text-(--muted)">
            <ul className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <Legend swatch="bg-(--line-strong)" text="removed channels" />
              <Legend swatch="bg-(--snake)" text="kept channels" />
              <Legend swatch="bg-int8" text="after INT8" />
            </ul>
            <button
              type="button"
              onClick={() => play(true)}
              className="inline-flex h-8 items-center gap-1.5 rounded-full bg-(--chip) px-3 text-(--chip-fg) transition-colors hover:bg-(--chip-hover)"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              Replay
            </button>
          </div>

          <ModelStrip
            layers={LAYERS}
            cut={cut}
            quant={quant}
            label={`ResNet-20, ${LAYERS.length} layers: channels kept after a 50% structured prune, then quantized to INT8.`}
            className="mt-6 h-[clamp(150px,24vw,300px)] w-full"
          />

          <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-2xl bg-(--line) lg:grid-cols-4">
            <Stat label="Parameters" hint={`from ${count(STATS.params_before)}`}>
              <MotionNumber value={params} format={fmtCount} initial={STATS.params_before} />
            </Stat>
            <Stat label="Size" hint={`from ${mb(BASE.size_mb)}`}>
              <MotionNumber value={size} format={fmtMb} initial={BASE.size_mb} />
            </Stat>
            <Stat label="Accuracy" hint={`${MODEL.totals.accuracy_drop_pp.toFixed(2)} pp lower than ${pct(BASE.accuracy)}`}>
              <MotionNumber value={acc} format={fmtPct} initial={BASE.accuracy} />
            </Stat>
            <Stat label="Median latency" hint={`from ${ms(BASE.median_ms)}, Apple M5`}>
              <MotionNumber value={lat} format={fmtMs} initial={BASE.median_ms} />
            </Stat>
          </dl>
          <p className="mt-4 text-[12px] leading-snug text-(--muted)">
            ResNet-20, CIFAR-10 test set, {MODEL.schedule}. Measured on {MODEL.provenance.host}. The counters move
            between the baseline and the final recorded measurement.
          </p>
        </div>
      </div>
    </Band>
  );
}

function Legend({ swatch, text }: { swatch: string; text: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={`h-2.5 w-2.5 rounded-[3px] ${swatch}`} aria-hidden="true" />
      {text}
    </li>
  );
}

function Stat({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="bg-(--bg) p-4 sm:p-5">
      <dt className="label-xs !text-(--muted)">{label}</dt>
      <dd className="mt-2 text-[clamp(22px,2.6vw,34px)] leading-none tracking-[-0.03em]">{children}</dd>
      <p className="mt-2 text-[12px] leading-snug text-(--muted)">{hint}</p>
    </div>
  );
}

