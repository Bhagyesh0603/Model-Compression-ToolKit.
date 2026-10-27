import type { Metadata } from 'next';
import Link from 'next/link';
import { Code } from '@/src/site/Code';
import { Band } from '@/src/motion/Band';
import { PageFrame } from '@/src/site/PageFrame';
import { Section, Wrap } from '@/src/site/Section';

export const metadata: Metadata = { title: 'Run it — compresskit' };

export default function RunPage() {
  return (
    <PageFrame>
      <Band tone="light">
        <Wrap className="pb-16 pt-24 sm:pb-20 sm:pt-32">
          <p className="label-xs">Run it</p>
          <h1 className="mt-4 max-w-[18ch] text-balance text-[clamp(36px,4.8vw,66px)] font-normal leading-[1.03] tracking-[-0.05em]">
            From a trained CNN to an ONNX file
          </h1>
          <p className="mt-6 max-w-[60ch] text-[17px] leading-[1.65] text-(--muted)">
            Everything here is a command the repository provides. The quickest route is the Kaggle notebook; the same
            flow runs locally from the command line.
          </p>
        </Wrap>
      </Band>

      <Section
        title="On Kaggle, one notebook"
        lede={
          <p>
            Import the notebook, set the accelerator to a GPU and Internet to On, edit the config cell, Run All. It
            trains a baseline if needed, prunes gradually with fine-tuning, quantizes, evaluates the exported file and
            writes the ONNX file and an HTML report. See <Link href="/notebooks" className="underline underline-offset-4">the notebooks</Link>.
          </p>
        }
      />

      <Section title="Locally, from the command line">
        <div className="space-y-5">
          <Code label="Install">{`git clone https://github.com/Bhagyesh0603/Model-Compression-ToolKit..git
cd Model-Compression-ToolKit.
pip install -e .
compresskit probe          # what this CPU can do, including whether INT8 is fast`}</Code>
          <Code label="Train a baseline (skipped if the checkpoint exists)">{`compresskit train --arch resnet20 --skip-if-exists`}</Code>
          <Code label="Explicit pipeline: gradual pruning with fine-tuning, then quantize">{`compresskit pipeline --model checkpoints/resnet20_fp32.pt \\
    --prune 0.5 --steps 5 --fine-tune-epochs 5 --loss dkd \\
    --report report.html --json results.json`}</Code>
          <Code label="Or state a budget and let it decide">{`compresskit compress --model checkpoints/resnet20_fp32.pt \\
    --budget "accuracy_drop<1%,size<0.4mb" --report report.html`}</Code>
          <Code label="Check the result on the machine you will deploy to">{`compresskit bench artifacts/pipeline/3_quantize.onnx
python experiments/verify_onnx.py --model checkpoints/resnet20_fp32.pt --onnx <final>.onnx`}</Code>
        </div>
      </Section>

      <Section
        title="Your own data and model"
        lede={
          <p>
            Datasets are CIFAR-10, the Human Action Recognition set, or any image folder with one sub-directory per
            class. A checkpoint must be one the toolkit saved, because compressed models store the whole module.
          </p>
        }
      >
        <Code>{`compresskit train --arch har_resnet18 --dataset folder --data-root /path/to/images --input-size 224
compresskit pipeline --model checkpoints/har_resnet18_fp32.pt --dataset folder \\
    --data-root /path/to/images --input-size 224 --prune 0.5 --steps 5 --fine-tune-epochs 5`}</Code>
      </Section>

      <Section
        title="What it does not do"
        lede={
          <ul className="list-disc space-y-1.5 pl-5">
            <li>PyTorch image-classification models only. No TensorFlow, and no detection or segmentation models.</li>
            <li>Tested on ResNet and VGG-style networks. Depthwise or grouped convolutions (MobileNet-style) are untested for pruning.</li>
            <li>The model must export to ONNX, since every final measurement is made on that file.</li>
            <li>Latency from a cloud CPU describes that CPU. Measure on your target with compresskit bench.</li>
          </ul>
        }
      />
    </PageFrame>
  );
}
