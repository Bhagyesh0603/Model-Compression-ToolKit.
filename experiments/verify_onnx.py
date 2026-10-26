"""Does the compressed .onnx actually work? Run it next to the original model.

    python experiments/verify_onnx.py --model checkpoints/resnet20_fp32.pt \
        --onnx artifacts/demo/3_quantize.onnx

Runs both models on the same real test images and reports accuracy, how often
they agree, file size and latency, then prints a few side-by-side predictions.
Works for any dataset the CLI knows (--dataset / --data-root as in `compresskit`).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compresskit.core.benchmark import benchmark, make_session  # noqa: E402
from compresskit.core.export import size_mb  # noqa: E402
from compresskit.zoo.datasets import get_bundle  # noqa: E402
from compresskit.zoo.train import load_model  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="original FP32 checkpoint")
    parser.add_argument("--onnx", required=True, help="compressed model from the pipeline")
    parser.add_argument("--dataset", help="default: what the checkpoint was trained on")
    parser.add_argument("--data-root")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--images", type=int, default=1000, help="how many test images to use")
    parser.add_argument("--show", type=int, default=8, help="side-by-side examples to print")
    args = parser.parse_args(argv)

    original, meta = load_model(args.model)
    bundle = get_bundle(args.dataset or meta.get("dataset", "cifar10"), args.data_root, args.input_size)
    names = bundle.class_names

    session = make_session(args.onnx, threads=0)
    input_name = session.get_inputs()[0].name

    truth, orig_pred, comp_pred = [], [], []
    with torch.no_grad():
        for images, targets in bundle.test_loader(batch_size=64):
            orig_pred += original(images).argmax(1).tolist()
            comp_pred += session.run(None, {input_name: images.numpy().astype(np.float32)})[0].argmax(1).tolist()
            truth += targets.tolist()
            if len(truth) >= args.images:
                break

    truth, orig_pred, comp_pred = (np.array(a[: args.images]) for a in (truth, orig_pred, comp_pred))
    n = len(truth)
    orig_acc, comp_acc = (orig_pred == truth).mean(), (comp_pred == truth).mean()
    agree = (orig_pred == comp_pred).mean()

    print(f"\nTested on {n} real test images ({bundle.name})\n")
    print(f"{'':22}{'original (PyTorch)':>20}{'compressed (ONNX)':>20}")
    print(f"{'accuracy':22}{orig_acc * 100:>19.2f}%{comp_acc * 100:>19.2f}%")
    print(f"{'agrees with original':22}{'':>20}{agree * 100:>19.2f}%")
    original_size = Path(args.model).stat().st_size / 1e6
    print(f"{'file size':22}{original_size:>17.2f} MB{size_mb(args.onnx):>17.2f} MB")
    latency = benchmark(args.onnx, bundle.input_shape)
    print(f"{'latency (this CPU)':22}{'':>20}{latency.median_ms:>17.3f} ms")

    print("\nSide by side (✓ = correct):")
    print(f"{'#':>3}  {'true':14}{'original':16}{'compressed':16}")
    for i in range(min(args.show, n)):
        tag = lambda p: f"{names[p]} {'✓' if p == truth[i] else '✗'}"  # noqa: E731
        print(f"{i:>3}  {names[truth[i]]:14}{tag(orig_pred[i]):16}{tag(comp_pred[i]):16}")

    ok = comp_acc >= orig_acc - 0.05 and agree >= 0.85
    print(f"\n{'PASS' if ok else 'CHECK'}: compressed model "
          f"{'works — close to the original' if ok else 'differs a lot from the original'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
