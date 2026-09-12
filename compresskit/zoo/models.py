import torch.nn as nn
import torch.nn.functional as F

CIFAR10_CLASSES = 10


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class CifarResNet(nn.Module):
    """CIFAR-style ResNet (He et al. 2016): 6n+2 layers, stages of 16/32/64 channels."""

    def __init__(self, blocks_per_stage, num_classes=CIFAR10_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes = 16
        self.layer1 = self._make_stage(16, blocks_per_stage, 1)
        self.layer2 = self._make_stage(32, blocks_per_stage, 2)
        self.layer3 = self._make_stage(64, blocks_per_stage, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride if i == 0 else 1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class ResNet18Cifar(nn.Module):
    """torchvision-style ResNet-18 (4 stages x 2 BasicBlocks, 64/128/256/512),

    with the ImageNet stem (7x7 stride-2 conv + maxpool, built for 224x224
    input) swapped for a 3x3 stride-1 conv and no maxpool. Without this swap
    a 32x32 CIFAR image is downsampled to 1x1 before it reaches stage 2 — the
    standard adaptation used by every CIFAR-ResNet-18 reference implementation.
    """

    def __init__(self, num_classes=CIFAR10_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_planes = 64
        self.layer1 = self._make_stage(64, 2, 1)
        self.layer2 = self._make_stage(128, 2, 2)
        self.layer3 = self._make_stage(256, 2, 2)
        self.layer4 = self._make_stage(512, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride if i == 0 else 1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]


class CifarVGG(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes=CIFAR10_CLASSES):
        super().__init__()
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


def resnet56(num_classes=CIFAR10_CLASSES):
    return CifarResNet(blocks_per_stage=9, num_classes=num_classes)


def resnet20(num_classes=CIFAR10_CLASSES):
    return CifarResNet(blocks_per_stage=3, num_classes=num_classes)


def vgg16_bn(num_classes=CIFAR10_CLASSES):
    return CifarVGG(num_classes=num_classes)


def resnet18(num_classes=CIFAR10_CLASSES):
    return ResNet18Cifar(num_classes=num_classes)


ARCHITECTURES = {
    "resnet56": resnet56,
    "resnet20": resnet20,
    "resnet18": resnet18,
    "vgg16_bn": vgg16_bn,
}


def build(name, num_classes=CIFAR10_CLASSES):
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](num_classes=num_classes)
