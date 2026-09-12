"""Before/after compression test for the HAR-fine-tuned ResNet-18.

This is the toolkit's first real-world (non-CIFAR) subject: a genuine
ImageNet-pretrained torchvision resnet18, fine-tuned to 15 human-action
classes on 224x224 images — not a 32x32 toy. input_shape is threaded through
explicitly everywhere below because compresskit's default is (1,3,32,32).

    python experiments/compress_har.py --model checkpoints/har_resnet18_fp32.pt
    python experiments/compress_har.py --model checkpoints/har_resnet18_fp32.pt --ratio 0.5 --distill 5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compresskit.har import calibration_loader, test_loader, train_loader  # noqa: E402
from compresskit.host import probe  # noqa: E402
from compresskit.pipeline import Pipeline  # noqa: E402
from compresskit.report import render  # noqa: E402
from compresskit.sensitivity import analyze  # noqa: E402
from compresskit.train import load_model  # noqa: E402

INPUT_SHAPE = (1, 3, 224, 224)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="checkpoints/har_resnet18_fp32.pt",
                        help="checkpoint from experiments/import_har_checkpoint.py")
    parser.add_argument("--ratio", type=float, default=0.3, help="pruning ratio")
    parser.add_argument("--align", type=int, default=16)
    parser.add_argument("--precision", default="auto", choices=["auto", "int8", "fp16"])
    parser.add_argument("--distill", type=int, default=0, help="distillation epochs (0 = skip)")
    parser.add_argument("--loss", default="dkd", choices=["kd", "dkd"])
    parser.add_argument("--report", default="artifacts/har_resnet18_report.html")
    args = parser.parse_args(argv)

    host = probe()
    print(host.describe(), "\n")

    model, meta = load_model(args.model)
    print(f"baseline: {meta.get('arch', 'har_resnet18')} "
          f"@ {meta.get('accuracy', 0) * 100:.2f}% top-1 (as measured at import time)\n")

    calib = calibration_loader(n=512, batch_size=32)
    test = test_loader(batch_size=64)
    train = train_loader(batch_size=64) if args.distill else None

    print("per-layer sensitivity (224x224 forward passes — this step is the slow one):")
    sensitivity = analyze(model, calib, batches=4)
    print(sensitivity.summary(top=5), "\n")

    pipe = Pipeline(model, calib, test, train_loader=train,
                    input_shape=INPUT_SHAPE, workdir="artifacts/har_resnet18")

    if args.ratio > 0:
        pipe.prune(ratio=args.ratio, align=args.align,
                  ratio_overrides=sensitivity.prune_ratios(args.ratio)).recover()
    if args.distill:
        pipe.distill(epochs=args.distill, loss=args.loss)
    pipe.quantize(precision=args.precision)

    result = pipe.run(verbose=True)

    print("\n=== before -> after ===")
    print(f"baseline : {result.baseline.summary()}")
    print(f"final    : {result.final.summary()}")
    speedup = result.baseline.latency.median_ms / result.final.latency.median_ms
    drop_pp = (result.baseline.accuracy - result.final.accuracy) * 100
    print(f"speedup  : {speedup:.2f}x")
    print(f"acc drop : {drop_pp:+.2f}pp")
    print(f"size     : {result.baseline.size_mb:.2f} MB -> {result.final.size_mb:.2f} MB")

    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(f"  ! {w}")

    report_path = render(result, args.report, sensitivity=sensitivity,
                         title="HAR ResNet-18 — before/after compression")
    print(f"\nreport -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
