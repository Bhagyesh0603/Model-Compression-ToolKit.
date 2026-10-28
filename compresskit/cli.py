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


PRETRAINED_ARCHS = ("har_resnet18", "resnet18_pretrained")


def _is_pretrained(arch: str) -> bool:
    from .zoo.torchvision_models import is_torchvision_arch

    return arch in PRETRAINED_ARCHS or is_torchvision_arch(arch)


def _dataset_name(args, meta=None) -> str:
    """Explicit --dataset, else what the checkpoint was trained on, else a guess from --arch."""
    if getattr(args, "dataset", None):
        return args.dataset
    if meta and meta.get("dataset"):
        return meta["dataset"]
    from .zoo.torchvision_models import is_torchvision_arch

    if is_torchvision_arch(args.arch):
        return "imagenet"
    return "har" if args.arch in PRETRAINED_ARCHS else "cifar10"


def _bundle(args, meta=None):
    from .zoo.datasets import get_bundle

    return get_bundle(_dataset_name(args, meta), getattr(args, "data_root", None),
                      getattr(args, "input_size", 224), getattr(args, "test_samples", None),
                      getattr(args, "holdout_val", 0))


def _loaders(args, bundle):
    batch = args.batch_size or (128 if bundle.name == "cifar10" else 64)
    return (
        bundle.calibration_loader(n=args.calib, batch_size=32 if bundle.name != "cifar10" else 64),
        bundle.test_loader(batch_size=batch),
        bundle.train_loader(batch_size=batch),
    )


def _build_arch(arch: str, num_classes: int):
    from .zoo.har import build_har_resnet18
    from .zoo.models import build
    from .zoo.torchvision_models import IMAGENET_CLASSES, build_torchvision, is_torchvision_arch, replace_head

    if is_torchvision_arch(arch):
        model = build_torchvision(arch)
        # A different class count means a new, untrained head: train it with `compresskit train`.
        return model if num_classes == IMAGENET_CLASSES else replace_head(model, num_classes)
    if arch in PRETRAINED_ARCHS:
        return build_har_resnet18(num_classes=num_classes, pretrained=True)
    return build(arch, num_classes=num_classes)


def _load_model(args):
    """Returns (model, meta). Without --model, an untrained --arch (numbers are meaningless)."""
    from .zoo.train import load_model

    if args.model:
        model, meta = load_model(args.model)
        accuracy = meta.get("accuracy")
        print(f"loaded {args.model} ({meta.get('arch', '?')}"
              + (f", baseline accuracy {accuracy * 100:.2f}%)" if accuracy is not None else ")"))
        return model, meta
    if _is_pretrained(args.arch):
        print(f"no --model given; using {args.arch} with its pretrained ImageNet weights")
    else:
        print(f"no --model given; using an untrained {args.arch} (numbers will be meaningless)")
    return _build_arch(args.arch, _bundle(args).num_classes), {}


def _report_extras(result, bundle, meta, test) -> dict:
    """Detail sections for the HTML report: training curve and per-class accuracy."""
    from .core.benchmark import evaluate_onnx

    extras = {"training_history": meta.get("history"), "class_names": bundle.class_names}
    try:
        extras["per_class"] = (evaluate_onnx(result.baseline.artifact, test),
                               evaluate_onnx(result.final.artifact, test))
    except Exception as error:  # a report section must never sink a finished run
        print(f"  (per-class accuracy skipped: {error})")
    return extras


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
    from .zoo.train import TrainConfig, save_model, train

    from .zoo.torchvision_models import is_torchvision_arch

    if is_torchvision_arch(args.arch) and _dataset_name(args) == "imagenet":
        print(f"{args.arch} is already trained (ImageNet weights); compress it directly with "
              f"`compresskit pipeline --arch {args.arch} --dataset imagenet --data-root ...`", file=sys.stderr)
        return 2
    out = Path(args.out or f"checkpoints/{args.arch}_fp32.pt")
    if args.skip_if_exists and out.exists():
        print(f"{out} already exists — skipping training (--skip-if-exists)")
        return 0

    bundle = _bundle(args)
    pretrained = _is_pretrained(args.arch)
    epochs = args.epochs or (10 if pretrained else 60)
    lr = args.lr or (0.01 if pretrained else 0.1)
    batch = args.batch_size or (128 if bundle.name == "cifar10" else 64)
    tl, test = bundle.train_loader(batch_size=batch), bundle.test_loader(batch_size=batch)
    # With a held-out split, the per-epoch curve is read on it, so the test set is
    # only looked at once, at the end.
    vl = bundle.val_loader(batch_size=batch) if bundle.val_loader else test
    model = _build_arch(args.arch, bundle.num_classes)

    if pretrained and not is_torchvision_arch(args.arch):
        # Same recipe as the original notebook: keep the ImageNet backbone
        # frozen and fine-tune only layer4 and the new head.
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("layer4", "fc"))

    history = []

    def on_epoch(_, epoch, stats):
        history.append({k: stats[k] for k in ("epoch", "loss", "train_acc", "val_acc") if k in stats})
        print(f"epoch {epoch + 1:3d}/{epochs}  loss {stats['loss']:.3f}  "
              f"train {stats['train_acc'] * 100:5.2f}%  val {stats.get('val_acc', 0) * 100:5.2f}%  "
              f"{stats['seconds']:.1f}s", flush=True)

    train(model, tl, TrainConfig(epochs=epochs, lr=lr), val_loader=vl, on_epoch=on_epoch)
    accuracy = accuracy_torch(model, test)
    path = save_model(model, out, {
        "arch": args.arch, "epochs": epochs, "accuracy": accuracy, "dataset": bundle.name,
        "input_shape": list(bundle.input_shape), "class_names": bundle.class_names, "history": history,
        "holdout_val": getattr(args, "holdout_val", 0),
    })
    print(f"final top-1 {accuracy * 100:.2f}% -> {path}")
    return 0


def cmd_sensitivity(args) -> int:
    from .sensitivity import analyze

    model, meta = _load_model(args)
    bundle = _bundle(args, meta)
    report = analyze(model, bundle.calibration_loader(n=args.calib, batch_size=32), batches=args.batches)
    print()
    print(report.summary(top=args.top))
    print()
    print(f"most sensitive: {report.most_sensitive(2)}")
    print("  -> exclude these from INT8, or prune them less than the rest")
    return 0


def cmd_pipeline(args) -> int:
    from .pipeline import Pipeline
    from .report import render

    model, meta = _load_model(args)
    bundle = _bundle(args, meta)
    calib, test, train = _loaders(args, bundle)

    pipe = Pipeline(model, calib, test, train_loader=train, input_shape=bundle.input_shape,
                    workdir=args.workdir)
    if args.prune:
        pipe.prune(
            ratio=args.prune,
            align=args.align,
            steps=args.steps,
            fine_tune_epochs=args.fine_tune_epochs,
            loss=args.loss,
        )
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

    if args.report:
        from .sensitivity import analyze

        sensitivity = analyze(model, calib, batches=4)
        written = render(result, args.report, sensitivity=sensitivity,
                         **_report_extras(result, bundle, meta, test))
        print(f"\nreport -> {written}")
    if args.json:
        print(f"json   -> {result.save_json(args.json)}")
    return 0


def cmd_compress(args) -> int:
    from .report import render_search
    from .search import compress

    model, meta = _load_model(args)
    bundle = _bundle(args, meta)
    calib, test, train = _loaders(args, bundle)
    select = bundle.val_loader(batch_size=256) if bundle.val_loader else None
    if select is not None:
        print("selection on the held-out validation split; the test set is only used to report the winner")

    result = compress(
        model, calib, test,
        select_loader=select,
        budget=args.budget,
        train_loader=train,
        input_shape=bundle.input_shape,
        workdir=args.workdir,
        align=args.align,
        distill_epochs=args.distill,
        distill_loss=args.loss,
        qat_epochs=args.qat_epochs,
        prune_steps=args.steps,
        prune_fine_tune_epochs=args.fine_tune_epochs,
        precision=args.precision,
        allow_slow_int8=args.allow_slow_int8,
    )
    print()
    print("=" * 78)
    print(result.summary())

    if args.report:
        extras = _report_extras(result.chosen.result, bundle, meta, test) if result.chosen else {}
        print(f"\nreport -> {render_search(result, args.report, **extras)}")
    if args.json and result.chosen:
        written = result.chosen.result.save_json(args.json)
        if result.test_accuracy:
            import json as _json
            payload = _json.loads(Path(written).read_text())
            payload["selection"] = {"split": "validation (held out of train)", "test_accuracy": result.test_accuracy}
            Path(written).write_text(_json.dumps(payload, indent=2))
        print(f"json   -> {written}")
    if result.test_accuracy:
        t = result.test_accuracy
        print(f"held-out TEST accuracy of the chosen configuration: baseline {t['baseline'] * 100:.2f}% -> "
              f"{t['final'] * 100:.2f}%  (selection used validation only)")
    return 0 if result.met else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compresskit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(sub):
        sub.add_argument("--dataset", choices=["cifar10", "cifar10-upscaled", "har", "folder", "imagenet"],
                         help="default: what the checkpoint was trained on, else guessed from --arch")
        sub.add_argument("--data-root", help="dataset location (har: folder with Training_set.csv; "
                                             "folder: one sub-directory per class; "
                                             "imagenet: train/ and val/ with 1000 class folders; "
                                             "cifar10-upscaled: CIFAR-10 at --input-size, for tv: models)")
        sub.add_argument("--input-size", type=int, default=224, help="image size for --dataset folder/imagenet")
        sub.add_argument("--holdout-val", type=int, default=0,
                         help="cifar10: hold this many training images out as a validation split (e.g. 5000); "
                              "the model must be trained with the same flag, or it has seen them")
        sub.add_argument("--test-samples", type=int, help="--dataset imagenet/cifar10-upscaled: score this many test "
                                                          "images (default 2000, 0 = all)")

    def add_model_args(sub, calib_default=512):
        sub.add_argument("--model", help="checkpoint from `compresskit train`")
        sub.add_argument("--arch", default="resnet20",
                         help="architecture if no --model (resnet20/56/18, vgg16_bn, har_resnet18, "
                              "or tv:<name> for any torchvision ImageNet model, e.g. tv:vgg16_bn)")
        sub.add_argument("--calib", type=int, default=calib_default, help="calibration samples (train split)")
        add_data_args(sub)

    probe = subparsers.add_parser("probe", help="report host CPU capabilities")
    probe.set_defaults(func=cmd_probe)

    bench = subparsers.add_parser("bench", help="benchmark an ONNX artifact")
    bench.add_argument("artifact")
    bench.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 32, 32])
    bench.add_argument("--warmup", type=int, default=50)
    bench.add_argument("--runs", type=int, default=200)
    bench.set_defaults(func=cmd_bench)

    train = subparsers.add_parser("train", help="train an FP32 baseline")
    train.add_argument("--arch", default="resnet20",
                       help="resnet20/56/18, vgg16_bn, or har_resnet18 (ImageNet-pretrained)")
    train.add_argument("--epochs", type=int, help="default 60 (10 for pretrained archs)")
    train.add_argument("--lr", type=float, help="default 0.1 (0.01 for pretrained archs)")
    train.add_argument("--batch-size", type=int, help="default 128 for CIFAR-10, else 64")
    train.add_argument("--out")
    train.add_argument("--skip-if-exists", action="store_true", help="do nothing if --out already exists")
    add_data_args(train)
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
    pipe.add_argument("--fine-tune-epochs", type=int, default=0, help="fine-tuning epochs per pruning step")
    pipe.add_argument("--distill", type=int, default=0, help="distillation epochs")
    pipe.add_argument("--loss", default="dkd", choices=["kd", "dkd"])
    pipe.add_argument("--precision", default="auto", choices=["auto", "int8", "fp16", "fp32"])
    pipe.add_argument("--allow-slow-int8", action="store_true")
    pipe.add_argument("--no-quantize", action="store_true")
    pipe.add_argument("--batch-size", type=int)
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
    auto.add_argument("--batch-size", type=int)
    auto.add_argument("--steps", type=int, default=1, help="gradual pruning steps in the recovery stage")
    auto.add_argument("--fine-tune-epochs", type=int, default=0, help="fine-tune epochs after each pruning step")
    auto.add_argument("--precision", default="auto", choices=["auto", "int8", "fp16", "fp32"])
    auto.add_argument("--allow-slow-int8", action="store_true")
    auto.add_argument("--json", help="write raw measurements here")
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
