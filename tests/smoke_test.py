from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mirfd.losses import MIRFDLoss
from mirfd.models import MIRFDNet


def main() -> None:
    model = MIRFDNet(
        in_channels=1,
        num_classes=1,
        base_dim=8,
        depths=(1, 1, 1, 1),
        use_aux_heads=True,
    )
    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.97).float()
    outputs = model(x, return_features=True)
    assert outputs["logits"].shape == (2, 1, 64, 64)
    criterion = MIRFDLoss(aux_weight=0.2, spectral_low_weight=0.01, spectral_high_weight=0.01)
    loss, details = criterion(outputs, y)
    loss.backward()
    print("smoke ok", details)


if __name__ == "__main__":
    main()
