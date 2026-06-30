from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mirfd.losses import MIRFDLoss
from mirfd.metrics import segmentation_metrics
from mirfd.models import MIRFDNet, build_model
from mirfd.utils import load_config


def run_basic_smoke() -> None:
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
    metrics = segmentation_metrics(outputs["logits"], y)
    assert "pd" in metrics and "fa" in metrics
    print("basic smoke ok", details)


def run_v2_config_smoke() -> None:
    cfg = deepcopy(load_config(ROOT / "configs" / "mirfd_nuaa_sirst_ss2d_v2.yaml"))

    cfg["model"]["base_dim"] = 8
    cfg["model"]["depths"] = [1, 1, 1, 1]
    cfg["model"]["use_aux_heads"] = False
    cfg["model"]["mamba"]["variant"] = "fallback"
    cfg["model"]["mamba"]["scan_backend"] = "ref"
    cfg["model"]["high_skip_stages"] = [1, 2]
    cfg["model"]["mirfd"]["gate_mode"] = "centered"
    cfg["model"]["mirfd"]["gate_scale_min"] = 0.25
    cfg["model"]["mirfd"]["gate_scale_max"] = 1.75
    cfg["model"]["mirfd"]["high_residual_mode"] = "add_scaled"
    cfg["model"]["mirfd"]["hfe_scale_init"] = 0.1
    cfg["loss"]["spectral_low_weight"] = 0.001
    cfg["loss"]["spectral_high_weight"] = 0.001
    cfg["loss"]["spectral_high_target"] = "high_raw"
    cfg["loss"]["gate_bg_weight"] = 0.01

    model = build_model(cfg)
    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.97).float()

    logits = model(x, return_dict=False)
    assert logits.shape == (2, 1, 64, 64)

    outputs = model(x, return_features=True)
    assert outputs["logits"].shape == (2, 1, 64, 64)
    for key in ("low0", "low", "residual", "high_raw", "high_hat", "gate"):
        assert key in outputs["features"]

    criterion = MIRFDLoss(**cfg["loss"])
    loss, details = criterion(outputs, y)
    loss.backward()
    assert "spectral_high" in details

    bad_stage_cfg = deepcopy(cfg)
    bad_stage_cfg["model"]["high_skip_stages"] = [1, 2, 4]
    try:
        build_model(bad_stage_cfg)
    except ValueError as exc:
        assert "Stage 4 high_hat" in str(exc)
    else:
        raise AssertionError("high_skip_stages containing stage 4 should be rejected")

    disabled_skip_cfg = deepcopy(cfg)
    disabled_skip_cfg["model"]["decoder"]["use_high_residual_skip"] = False
    disabled_skip_cfg["model"]["high_skip_stages"] = [1]
    try:
        build_model(disabled_skip_cfg)
    except ValueError as exc:
        assert "use_high_residual_skip=True" in str(exc)
    else:
        raise AssertionError("high_skip_stages should require decoder high residual skip")

    print("v2 config smoke ok", details)


def main() -> None:
    run_basic_smoke()
    run_v2_config_smoke()


if __name__ == "__main__":
    main()
