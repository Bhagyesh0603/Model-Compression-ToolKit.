"""The one-flow path: configurable dataset roots, the generic `folder` dataset,
and train -> gradual prune -> quantize -> detailed report through the CLI.

All data is synthetic and tiny; nothing is downloaded.
"""

import json

import numpy as np
import pytest
from PIL import Image

from compresskit import cli
from compresskit.zoo import har
from compresskit.zoo.datasets import get_bundle


@pytest.fixture
def image_folder(tmp_path):
    rng = np.random.default_rng(0)
    for label, name in enumerate(["red", "green", "blue"]):
        (tmp_path / "imgs" / name).mkdir(parents=True)
        for i in range(24):
            pixels = rng.integers(0, 60, (32, 32, 3), dtype=np.uint8)
            pixels[..., label] += 150
            Image.fromarray(pixels).save(tmp_path / "imgs" / name / f"{i:03d}.png")
    return tmp_path / "imgs"


def test_har_root_prefers_explicit_then_env(tmp_path, monkeypatch):
    monkeypatch.setenv(har.HAR_ROOT_ENV, str(tmp_path / "from_env"))
    assert har.resolve_root() == tmp_path / "from_env"
    assert har.resolve_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_folder_split_is_deterministic_and_disjoint(image_folder):
    a = get_bundle("folder", image_folder, input_size=32)
    b = get_bundle("folder", image_folder, input_size=32)
    assert a.class_names == ["blue", "green", "red"]
    assert a.input_shape == (1, 3, 32, 32)
    train, test = a.train_loader(8).dataset, a.test_loader(8).dataset
    assert set(train.indices).isdisjoint(test.indices)
    assert test.indices == b.test_loader(8).dataset.indices
    sub = a.calibration_loader(n=10).dataset  # Subset of positions within the train list
    calib_idx = {sub.dataset.indices[i] for i in sub.indices}
    assert calib_idx <= set(train.indices)  # calibration never touches the test split


def test_folder_dataset_requires_a_root():
    with pytest.raises(ValueError):
        get_bundle("folder", None)


@pytest.mark.slow
def test_train_pipeline_report_end_to_end(image_folder, tmp_path):
    ckpt = tmp_path / "ckpt" / "m.pt"
    common = ["--arch", "resnet20", "--dataset", "folder", "--data-root", str(image_folder),
              "--input-size", "32"]

    assert cli.main(["train", *common, "--epochs", "1", "--batch-size", "8", "--out", str(ckpt)]) == 0
    # a second run with --skip-if-exists must not retrain
    stamp = ckpt.stat().st_mtime_ns
    assert cli.main(["train", *common, "--out", str(ckpt), "--skip-if-exists"]) == 0
    assert ckpt.stat().st_mtime_ns == stamp

    report, raw = tmp_path / "report.html", tmp_path / "results.json"
    assert cli.main([
        "pipeline", "--model", str(ckpt), "--dataset", "folder", "--data-root", str(image_folder),
        "--input-size", "32", "--prune", "0.3", "--steps", "2", "--fine-tune-epochs", "1",
        "--precision", "int8", "--allow-slow-int8", "--batch-size", "8", "--calib", "16",
        "--workdir", str(tmp_path / "work"), "--report", str(report), "--json", str(raw),
    ]) == 0

    html = report.read_text()
    for section in ("Gradual pruning schedule", "Channels per layer", "Baseline training curve",
                    "Per-class accuracy", "compresskit bench"):
        assert section in html
    data = json.loads(raw.read_text())
    assert [row["step"] for row in data["prune_history"]] == [1, 2]
    assert list((tmp_path / "work").glob("*quantize.onnx"))


def test_cifar_root_prefers_explicit_then_env(tmp_path, monkeypatch):
    from compresskit.zoo import data

    monkeypatch.setenv(data.CIFAR_ROOT_ENV, str(tmp_path / "from_env"))
    assert data.resolve_root() == tmp_path / "from_env"
    assert data.resolve_root(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv(data.CIFAR_ROOT_ENV)
    assert data.resolve_root() in (data.DEFAULT_ROOT, *data._KAGGLE_INPUT.glob("*"))


def test_holdout_split_is_deterministic_disjoint_and_sized():
    from compresskit.zoo.data import holdout_indices

    val, fit = holdout_indices(5000)
    assert len(val) == 5000 and len(fit) == 45_000
    assert set(val).isdisjoint(fit) and sorted(val + fit) == list(range(50_000))
    assert holdout_indices(5000) == (val, fit)


def test_holdout_is_cifar_only():
    with pytest.raises(ValueError, match="cifar10"):
        get_bundle("folder", None, holdout_val=100)


def test_search_selects_on_validation_and_reports_test_separately(tmp_path):
    """The loader the search ranks on is not the one that scores the winner."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from compresskit.search import compress

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten(), torch.nn.Linear(8, 4)).eval()
    mk = lambda n: DataLoader(TensorDataset(torch.randn(n, 3, 8, 8), torch.randint(0, 4, (n,))), batch_size=8)  # noqa: E731
    seen = {"val": 0}

    class Counting(DataLoader):
        def __iter__(self):
            seen["val"] += 1
            return super().__iter__()

    val = Counting(TensorDataset(torch.randn(16, 3, 8, 8), torch.randint(0, 4, (16,))), batch_size=8)
    result = compress(model, mk(16), mk(16), budget="", train_loader=mk(16), input_shape=(1, 3, 8, 8),
                      workdir=tmp_path, select_loader=val, verbose=False, search_runs=5)
    assert seen["val"] > 0, "the search never read the validation loader"
    assert result.test_accuracy and set(result.test_accuracy) == {"baseline", "final"}
