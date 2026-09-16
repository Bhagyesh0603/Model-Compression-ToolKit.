"""Command line interface.

    compresskit probe                      what this CPU can do
    compresskit train                      train an FP32 baseline
    compresskit sensitivity                rank layers by quantization damage
    compresskit compress --budget "..."    budget-driven auto compression
    compresskit pipeline --prune 0.5 ...   explicit staged pipeline
    compresskit bench model.onnx           time an existing artifact
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


def _loaders(args):
    from .zoo.data import calibration_loader, test_loader, train_loader

    return (
        calibration_loader(n=args.calib, batch_size=64),
        test_loader(batch_size=256),
        train_loader(batch_size=args.batch_size),
    )


def _load_model(args):
    from .zoo.models import build
    from .zoo.train import load_model

    if args.model:
        model, meta = load_model(args.model)
        print(f"loaded {args.model} ({meta.get('arch', '?')}, "
              f"baseline accuracy {meta.get('accuracy', float('nan')) * 100:.2f}%)")
        return model
    print(f"no --model given; using an untrained {args.arch} (numbers will be meaningless)")
    return build(args.arch)


def cmd_probe(args) -> int:
    from .core.host import probe

    print(probe().describe())
    return 0


def cmd_bench(args) -> int:
    from .core.benchmark import benchmark
    from .core.export import size_mb

    latency = benchmark(args.artifact, tuple(args.input_shape), warmup=args.warmup, runs=args.runs)
    print(f"{args.artifact}  {size_mb(args.artifact):.2f} MB")
    print(f"  median {latency.median_ms:.3f} ms   p95 {latency.p95_ms:.3f} ms   "
          f"mean {latency.mean_ms:.3f} +/- {latency.stdev_ms:.3f}")
    print(f"  ({latency.runs} runs, {latency.warmup} warmup, batch {latency.batch}, "
          f"{latency.threads} thread)")
    return 0


def cmd_train(args) -> int:
    from .core.benchmark import accuracy_torch
    from .zoo.data import test_loader, train_loader
    from .zoo.models import build
    from .zoo.train import TrainConfig, save_model, train

    tl, vl = train_loader(batch_size=args.batch_size), test_loader(batch_size=256)
    model = build(args.arch)

    def on_epoch(_, epoch, stats):
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {stats['loss']:.3f}  "
              f"train {stats['train_acc'] * 100:5.2f}%  val {stats.get('val_acc', 0) * 100:5.2f}%  "
              f"{stats['seconds']:.1f}s", flush=True)

    train(model, tl, TrainConfig(epochs=args.epochs, lr=args.lr), val_loader=vl, on_epoch=on_epoch)
    accuracy = accuracy_torch(model, vl)
    path = save_model(model, args.out or f"checkpoints/{args.arch}_fp32.pt",
                      {"arch": args.arch, "epochs": args.epochs, "accuracy": accuracy, "dataset": "cifar10"})
    print(f"final top-1 {accuracy * 100:.2f}% -> {path}")
    return 0


def cmd_sensitivity(args) -> int:
    from .sensitivity import analyze
    from .zoo.data import calibration_loader

    model = _load_model(args)
    report = analyze(model, calibration_loader(n=args.calib, batch_size=64), batches=args.batches)
    print()
    print(report.summary(top=args.top))
    print()
    print(f"most sensitive: {report.most_sensitive(2)}")
    print("  -> exclude these from INT8, or prune them less than the rest")
    return 0


def cmd_pipeline(args) -> int:
    from .pipeline import Pipeline
    from .report import render

    model = _load_model(args)
    calib, test, train = _loaders(args)

    pipe = Pipeline(model, calib, test, train_loader=train, workdir=args.workdir)
    if args.prune:
        pipe.prune(ratio=args.prune, align=args.align, steps=args.steps)
        pipe.recover()
    if args.distill:
        pipe.distill(epochs=args.distill, loss=args.loss)
    if not args.no_quantize:
        pipe.quantize(precision=args.precision, allow_slow_int8=args.allow_slow_int8)

    print(pipe.preflight())
    print()
    result = pipe.run()
    print()
    print(result.summary())

    sensitivity = None
    if args.report:
        from .sensitivity import analyze

        sensitivity = analyze(model, calib, batches=4)
        print(f"\nreport -> {render(result, args.report, sensitivity=sensitivity)}")
    if args.json:
        print(f"json   -> {result.save_json(args.json)}")
    return 0


def cmd_compress(args) -> int:
    from .report import render_search
    from .search import compress

    model = _load_model(args)
    calib, test, train = _loaders(args)

    result = compress(
        model, calib, test,
        budget=args.budget,
        train_loader=train,
        workdir=args.workdir,
        align=args.align,
        distill_epochs=args.distill,
        distill_loss=args.loss,
        qat_epochs=args.qat_epochs,
    )
    print()
    print("=" * 78)
    print(result.summary())

    if args.report:
        print(f"\nreport -> {render_search(result, args.report)}")
    return 0 if result.met else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compresskit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_model_args(sub, calib_default=512):
        sub.add_argument("--model", help="checkpoint from `compresskit train`")
        sub.add_argument("--arch", default="resnet20", help="architecture if no --model")
        sub.add_argument("--calib", type=int, default=calib_default, help="calibration samples (train split)")

    probe = subparsers.add_parser("probe", help="report host CPU capabilities")
    probe.set_defaults(func=cmd_probe)

    bench = subparsers.add_parser("bench", help="benchmark an ONNX artifact")
    bench.add_argument("artifact")
    bench.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 32, 32])
    bench.add_argument("--warmup", type=int, default=50)
    bench.add_argument("--runs", type=int, default=200)
    bench.set_defaults(func=cmd_bench)

    train = subparsers.add_parser("train", help="train an FP32 baseline on CIFAR-10")
    train.add_argument("--arch", default="resnet20")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--lr", type=float, default=0.1)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--out")
    train.set_defaults(func=cmd_train)

    sens = subparsers.add_parser("sensitivity", help="rank layers by quantization damage")
    add_model_args(sens)
    sens.add_argument("--batches", type=int, default=4)
    sens.add_argument("--top", type=int, default=20)
    sens.set_defaults(func=cmd_sensitivity)

    pipe = subparsers.add_parser("pipeline", help="run an explicit staged pipeline")
    add_model_args(pipe)
    pipe.add_argument("--prune", type=float, default=0.0, help="structured pruning ratio")
    pipe.add_argument("--align", type=int, default=16, help="SIMD channel alignment")
    pipe.add_argument("--steps", type=int, default=1, help="iterative pruning steps")
    pipe.add_argument("--distill", type=int, default=0, help="distillation epochs")
    pipe.add_argument("--loss", default="dkd", choices=["kd", "dkd"])
    pipe.add_argument("--precision", default="auto", choices=["auto", "int8", "fp16", "fp32"])
    pipe.add_argument("--allow-slow-int8", action="store_true")
    pipe.add_argument("--no-quantize", action="store_true")
    pipe.add_argument("--batch-size", type=int, default=128)
    pipe.add_argument("--workdir", default="artifacts/pipeline")
    pipe.add_argument("--report", help="write an HTML report here")
    pipe.add_argument("--json", help="write raw measurements here")
    pipe.set_defaults(func=cmd_pipeline)

    auto = subparsers.add_parser("compress", help="budget-driven automatic compression")
    add_model_args(auto)
    auto.add_argument("--budget", default="", help='e.g. "latency<1ms,accuracy_drop<1%%"')
    auto.add_argument("--align", type=int, default=16)
    auto.add_argument("--distill", type=int, default=10, help="distillation epochs if needed")
    auto.add_argument("--loss", default="dkd", choices=["kd", "dkd"])
    auto.add_argument("--qat-epochs", type=int, default=3, help="QAT epochs if quantize-only misses accuracy")
    auto.add_argument("--batch-size", type=int, default=128)
    auto.add_argument("--workdir", default="artifacts/search")
    auto.add_argument("--report", help="write an HTML report here")
    auto.set_defaults(func=cmd_compress)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
