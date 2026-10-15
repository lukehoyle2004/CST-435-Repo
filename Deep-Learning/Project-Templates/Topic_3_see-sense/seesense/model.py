"""CifarCNN: the VGG-style network both See-Sense models share.

    input 3x32x32
    stage 1: [conv3x3(32)  -> (BN) -> ReLU] x2 -> MaxPool 2x2 -> Dropout2d   => 32x16x16
    stage 2: [conv3x3(64)  -> (BN) -> ReLU] x2 -> MaxPool 2x2 -> Dropout2d   => 64x8x8
    stage 3: [conv3x3(128) -> (BN) -> ReLU] x2 -> MaxPool 2x2 -> Dropout2d   => 128x4x4
    head:    global average pool -> Dropout -> Linear(128, 10)

The baseline turns off BatchNorm and sets every dropout to 0 (the Dropout2d
modules stay, so the layer layout and the Grad-CAM hook point are identical).
"""

import json
from pathlib import Path

import torch
import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int, batchnorm: bool = True) -> list[nn.Module]:
    # bias=False: BatchNorm subtracts the channel mean, which would cancel a bias anyway.
    if not batchnorm:
        return [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
    return [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)]


class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10, widths=(32, 64, 128),
                 dropout2d=(0.05, 0.1, 0.15), dropout_fc=0.3, batchnorm=True):
        super().__init__()
        stages, in_ch = [], 3
        for out_ch, p in zip(widths, dropout2d):
            stages.append(nn.Sequential(
                *conv_bn_relu(in_ch, out_ch, batchnorm),
                *conv_bn_relu(out_ch, out_ch, batchnorm),
                nn.MaxPool2d(2),
                nn.Dropout2d(p),
            ))
            in_ch = out_ch
        self.stage1, self.stage2, self.stage3 = stages
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_fc),
            nn.Linear(in_ch, num_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Last convolutional feature map before pooling (128 x 8 x 8): the Grad-CAM layer."""
        x = self.stage2(self.stage1(x))
        return self.stage3[:-2](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage3(self.stage2(self.stage1(x))))


def build_model(config: dict) -> CifarCNN:
    """The improved CNN, or the plain baseline when the run was trained with --baseline."""
    if config.get("baseline"):
        return CifarCNN(widths=config["widths"], dropout2d=(0.0, 0.0, 0.0), dropout_fc=0.0,
                        batchnorm=False)
    return CifarCNN(widths=config["widths"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_exported(stem: str | Path) -> tuple[CifarCNN, dict]:
    """Load `<stem>.pt` (a plain state_dict) and `<stem>.json` (the run config)."""
    stem = Path(stem)
    config = json.loads(stem.with_suffix(".json").read_text())
    model = build_model(config)
    model.load_state_dict(torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True))
    model.eval()
    return model, config
