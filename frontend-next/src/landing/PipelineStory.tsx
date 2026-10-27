'use client';

import { AnimatePresence, motion, useTransform, type MotionValue } from 'motion/react';
import { bnRecal, count, finalStage, mb, models, ms, pct, verify } from '@/src/data/results';
import { Band } from '@/src/motion/Band';
import { MotionNumber } from '@/src/motion/MotionNumber';
import { Scrolly, type ScrollyState, type ScrollyStep } from '@/src/motion/Scrolly';
import { DUR, EASE, windowed } from '@/src/motion/tokens';
import { AnimSawtooth } from '@/src/viz/AnimSawtooth';
import { ModelStrip } from '@/src/viz/ModelStrip';
import { layersOf } from '@/src/viz/layers';

const MODEL = models.find((m) => m.arch === 'resnet20')!;
const LAYERS = layersOf(MODEL)!;
const BASE = MODEL.baseline;
const PRUNED = MODEL.stages[1];
const FINAL = finalStage(MODEL);
const STATS = MODEL.prune_stats!;
const OPS = Object.values(MODEL.quantization?.quantized_ops ?? {}).reduce((a, b) => a + b, 0);

const STEPS: ScrollyStep[] = [
  {
    id: 'baseline',
    kicker: 'Baseline',
    title: 'Measure the original first.',
    body: (
      <p>
        The trained FP32 model is exported to ONNX and timed in ONNX Runtime: one thread, batch size 1, 50 warm-up runs
        and 200 timed runs. Everything later is compared with this, and the same model stays around as the teacher.
      </p>
    ),
  },
  {
    id: 'prune',
    kicker: 'Prune',
    title: 'Remove channels in steps, not at once.',
    body: (
      <p>
        Structured pruning deletes whole channels, so the matrices really shrink. {STATS.steps} cuts of the
        {' '}{pct(STATS.ratio_requested, 0)} target, each followed by fine-tuning. Accuracy drops at every cut and
        comes back after the fine-tune.
      </p>
    ),
  },
  {
    id: 'recover',
    kicker: 'Recover',
    title: 'Distil from the original, re-estimate BatchNorm.',
    body: (
      <p>
        Fine-tuning uses decoupled knowledge distillation against the unpruned model. BatchNorm statistics describe the
        old network, so they are re-estimated too: on a one-shot cut with no fine-tuning that alone moves accuracy from{' '}
        {pct(bnRecal.accuracy.after_cut_stale_bn)} to {pct(bnRecal.accuracy.after_bn_recalibration)}. Helpful, not
        enough.
      </p>
    ),
  },
  {
    id: 'quantize',
    kicker: 'Quantize',
    title: 'INT8 comes last, because it cannot be undone.',
    body: (
      <p>
        Integer weights cannot be pruned or trained further, so quantization is the final step. Here {OPS} of {OPS}{' '}
        compute operators received an INT8 kernel. On a CPU without fast integer instructions the toolkit defaults to
        FP16 instead.
      </p>
    ),
  },
  {
    id: 'verify',
    kicker: 'Verify',
    title: 'Run the delivered file, not the model in memory.',
    body: (
      <p>
        The compressed .onnx is tested next to the original on {count(verify.images)} real test images. It works and is
        close, but it is not identical, and the gap matches what the pipeline reported.
      </p>
    ),
  },
];

const N = STEPS.length;
const fmtCount = (v: number) => count(Math.round(v));
const fmtPct = (v: number) => pct(v);
const fmtMb = (v: number) => mb(v);
const fmtMs = (v: number) => ms(v);

/** The pinned pipeline story: one strip and four numbers that move through the stages. */
export function PipelineStory() {
  return (
    <Band id="pipeline" tone="dark" labelledBy="pipeline-title">
      <Scrolly
        steps={STEPS}
        label="Pipeline stages"
        header={
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="label-xs !text-(--muted)">The pipeline</p>
              <h2 id="pipeline-title" className="mt-3 max-w-[22ch] text-balance text-[clamp(26px,3.2vw,46px)] font-normal leading-[1.05] tracking-[-0.045em]">
                One model, five stages, every one measured.
              </h2>
            </div>
            <p className="figures-tab text-[13px] text-(--muted)">ResNet-20 · CIFAR-10 · {MODEL.provenance.host}</p>
          </div>
        }
        visual={(s) => <PipelineVisual state={s} />}
      />
    </Band>
  );
}

function PipelineVisual({ state }: { state: ScrollyState }) {
  const { progress, step } = state;
  const cut = useTransform(progress, (p) => windowed(p, 1 / N, 2 / N));
  const quant = useTransform(progress, (p) => windowed(p, 3 / N, 4 / N));
  const k = [0, 1 / N, 3 / N, 4 / N, 1];
  const params = useTransform(progress, k, [STATS.params_before, STATS.params_before, STATS.params_after, STATS.params_after, STATS.params_after]);
  const size = useTransform(progress, k, [BASE.size_mb, BASE.size_mb, PRUNED.size_mb, FINAL.size_mb, FINAL.size_mb]);
  const acc = useTransform(progress, k, [BASE.accuracy, BASE.accuracy, PRUNED.accuracy, FINAL.accuracy, FINAL.accuracy]);
  const lat = useTransform(progress, k, [BASE.median_ms, BASE.median_ms, PRUNED.median_ms, FINAL.median_ms, FINAL.median_ms]);
  const saw = useTransform(progress, (p) => windowed(p, 1 / N, 3 / N));

  return (
    <div>
      <ModelStrip
        layers={LAYERS}
        cut={cut}
        quant={quant}
        label="ResNet-20 channel columns, pruned then quantized as you scroll."
        className="h-[clamp(150px,26vh,260px)] w-full"
      />
      <dl className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-2xl bg-(--line) sm:grid-cols-4">
        <Cell label="Parameters"><MotionNumber value={params} format={fmtCount} initial={STATS.params_before} /></Cell>
        <Cell label="Size"><MotionNumber value={size} format={fmtMb} initial={BASE.size_mb} /></Cell>
        <Cell label="Accuracy"><MotionNumber value={acc} format={fmtPct} initial={BASE.accuracy} /></Cell>
        <Cell label="Latency, M5"><MotionNumber value={lat} format={fmtMs} initial={BASE.median_ms} /></Cell>
      </dl>
      <div className="relative mt-6 h-[clamp(190px,30vh,280px)] rounded-2xl border border-(--line) p-5">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={step <= 2 ? 'saw' : step}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: DUR.micro, ease: EASE }}
            className="h-full"
          >
            <Detail step={step} saw={saw} />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-(--bg) p-4">
      <dt className="label-xs !text-(--muted)">{label}</dt>
      <dd className="mt-1.5 text-[clamp(18px,2vw,26px)] leading-none tracking-[-0.03em]">{children}</dd>
    </div>
  );
}

function Detail({ step, saw }: { step: number; saw: MotionValue<number> }) {
  if (step <= 2) {
    return (
      <div className="flex h-full flex-col">
        <p className="text-[12px] text-(--muted)">
          Accuracy through the {MODEL.prune_history.length} cuts. Open dot: right after a cut. Solid: after fine-tuning.
        </p>
        <AnimSawtooth
          baseline={BASE.accuracy}
          history={MODEL.prune_history}
          reveal={saw}
          summary="Accuracy falls after each pruning cut and recovers after each fine-tune."
          className="mt-1 min-h-0 w-full flex-1"
        />
      </div>
    );
  }
  if (step === 3) {
    return (
      <div className="flex h-full flex-col justify-center gap-3.5">
        <p className="text-[12px] text-(--muted)">Pruned FP32 against the same model after INT8, {MODEL.provenance.host} (fast INT8).</p>
        <Bar label={`Size ${mb(PRUNED.size_mb)} to ${mb(FINAL.size_mb)}`} value={FINAL.size_mb / PRUNED.size_mb} showValue={false} />
        <Bar label={`Median latency ${ms(PRUNED.median_ms)} to ${ms(FINAL.median_ms)}`} value={FINAL.median_ms / PRUNED.median_ms} showValue={false} />
        <Bar label={`Accuracy ${pct(PRUNED.accuracy)} to ${pct(FINAL.accuracy)}`} value={FINAL.accuracy} accent showValue={false} />
        <p className="text-[12px] text-(--muted)">
          INT8 itself moved accuracy by a few hundredths of a point; almost all of the accuracy cost belongs to pruning.
        </p>
      </div>
    );
  }
  return (
    <div className="flex h-full flex-col justify-center gap-4">
      <Bar label="Original, PyTorch" value={verify.original_accuracy} />
      <Bar label="Compressed, ONNX" value={verify.compressed_accuracy} accent />
      <p className="text-[12px] text-(--muted)">
        The two models give the same prediction on {pct(verify.agreement, 1)} of the {count(verify.images)} images.
        Measured on {verify.provenance.host}.
      </p>
    </div>
  );
}

function Bar({ label, value, accent, showValue = true }: { label: string; value: number; accent?: boolean; showValue?: boolean }) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-[13px]">
        <span className="text-(--muted)">{label}</span>
        {showValue ? <span className="figures-tab">{pct(value)}</span> : null}
      </div>
      <div className="mt-1.5 h-2 rounded-full bg-(--line)">
        <motion.div
          className={`h-full rounded-full ${accent ? 'bg-int8' : 'bg-(--snake)'}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: DUR.reveal, ease: EASE }}
        />
      </div>
    </div>
  );
}
