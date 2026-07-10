from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metrics(path: Path, metrics: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in metrics) + "\n")


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def disable_dropout(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
