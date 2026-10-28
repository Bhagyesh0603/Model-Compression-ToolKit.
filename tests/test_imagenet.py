"""ImageNet-scale support: torchvision `--arch tv:<name>` and the imagenet dataset bundle."""

import numpy as np
import pytest
from PIL import Image

from compresskit import cli
from compresskit.zoo.datasets import IMAGENET_CLASSES, get_bundle
from compresskit.zoo.torchvision_models import build_torchvision


def _write_layout(root, classes, per_class=1, splits=("train", "val")):
    rng = np.random.default_rng(0)
    for split in splits:
        for c in range(classes):
            d = root / split / f"n{c:08d}"
            d.mkdir(parents=True)
            for i in range(per_class):
                Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{i}.png")
    return root


@pytest.fixture(scope="module")
def imagenet_root(tmp_path_factory):
    return _write_layout(tmp_path_factory.mktemp("in1k"), IMAGENET_CLASSES, per_class=1)


def test_torchvision_arch_builds_a_1000_class_model_without_downloading():
    model = build_torchvision("tv:resnet18", pretrained=False)
    assert model(__import__("torch").randn(1, 3, 224, 224)).shape == (1, 1000)


def test_torchvision_arch_rejects_unknown_names():
    with pytest.raises(ValueError):
        build_torchvision("tv:not_a_model", pretrained=False)


def test_imagenet_bundle_subsets_validation_deterministically(imagenet_root):
    a = get_bundle("imagenet", imagenet_root, input_size=32, test_samples=10)
    b = get_bundle("imagenet", imagenet_root, input_size=32, test_samples=10)
    assert a.input_shape == (1, 3, 32, 32) and a.num_classes == 1000
    ia, ib = a.test_loader(5).dataset.indices, b.test_loader(5).dataset.indices
    assert ia == ib and len(ia) == 10 and len(set(ia)) == 10
    assert len(get_bundle("imagenet", imagenet_root, 32, test_samples=0).test_loader(5).dataset) == 1000


def test_imagenet_bundle_refuses_class_subsets(tmp_path):
    # With 3 of 1000 folders, label i would not be the pretrained head's class i.
    root = _write_layout(tmp_path, 3)
    with pytest.raises(ValueError, match="1000"):
        get_bundle("imagenet", root)


def test_imagenet_bundle_needs_train_and_val(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_bundle("imagenet", _write_layout(tmp_path, 3, splits=("train",)))
    with pytest.raises(ValueError):
        get_bundle("imagenet", None)


def test_tv_arch_defaults_to_imagenet_and_retargets_the_head(monkeypatch):
    from compresskit.zoo import torchvision_models as tvm

    monkeypatch.setattr(tvm, "build_torchvision", lambda arch, pretrained=True: tvm.models.get_model(
        arch[len(tvm.TV_PREFIX):], weights=None))
    args = cli.build_parser().parse_args(["pipeline", "--arch", "tv:vgg16_bn"])
    assert cli._dataset_name(args) == "imagenet"
    assert cli._build_arch("tv:vgg16_bn", 1000).classifier[6].out_features == 1000
    assert cli._build_arch("tv:vgg16_bn", 10).classifier[6].out_features == 10
    assert cli._build_arch("tv:resnet18", 10).fc.out_features == 10
    assert cli.main(["train", "--arch", "tv:vgg16_bn"]) == 2  # ImageNet weights: nothing to train


class FakeCifar:
    """Module-level so DataLoader workers (spawn) can pickle it."""
    classes = ["a"] * 10

    def __init__(self, root, train=True, transform=None, download=False):
        self.n, self.transform = (60 if train else 40), transform

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.transform(Image.fromarray(np.full((32, 32, 3), i % 255, np.uint8))), i % 10


def test_upscaled_cifar_bundle_shapes_and_deterministic_subset(tmp_path, monkeypatch):
    import torchvision.datasets as tvd

    monkeypatch.setattr(tvd, "CIFAR10", FakeCifar)
    a = get_bundle("cifar10-upscaled", tmp_path, input_size=64, test_samples=8)
    assert a.name == "cifar10-upscaled" and a.input_shape == (1, 3, 64, 64) and a.num_classes == 10
    ds = a.test_loader(4).dataset
    assert len(ds) == 8 and ds.indices == get_bundle("cifar10-upscaled", tmp_path, 64, 8).test_loader(4).dataset.indices
    images, targets = next(iter(a.train_loader(4)))
    assert images.shape == (4, 3, 64, 64)
