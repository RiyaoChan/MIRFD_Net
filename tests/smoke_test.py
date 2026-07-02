from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from mirfd.losses import MIRFDLoss
from mirfd.metrics import segmentation_metrics
from mirfd.models import FFCFrequencyResidualEnhancer, FrequencySelectiveResidualEnhancer, MIRFDNet, build_model
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
    cfg["model"]["decoder_high_source"] = "high_raw"
    cfg["model"]["stage1_high_enhancer_type"] = "identity"
    cfg["model"]["mirfd"]["block_fusion_high_source"] = "high_raw"
    cfg["model"]["mirfd"]["gate_mode"] = "none"
    cfg["model"]["mirfd"]["gate_scale_min"] = 0.25
    cfg["model"]["mirfd"]["gate_scale_max"] = 1.75
    cfg["model"]["mirfd"]["high_residual_mode"] = "hfe"
    cfg["model"]["mirfd"]["hfe_scale_init"] = 0.1
    cfg["model"]["mirfd"]["high_enhancer_type"] = "freq_window"
    cfg["model"]["mirfd"]["fsre_num_bands"] = 4
    cfg["model"]["mirfd"]["fsre_window_size"] = 8
    cfg["model"]["mirfd"]["fsre_gamma_init"] = 0.1
    cfg["loss"]["spectral_low_weight"] = 0.001
    cfg["loss"]["spectral_high_weight"] = 0.001
    cfg["loss"]["spectral_high_target"] = "high_raw"
    cfg["loss"]["gate_bg_weight"] = 0.01

    model = build_model(cfg)
    assert model.decoder_high_source == "high_raw"
    assert model.high_skip_stages == {1, 2}
    assert isinstance(model.stage1_hfe, nn.Identity)
    for stage in (model.stage2, model.stage3, model.stage4):
        for block in stage.blocks:
            assert isinstance(block.hfe, FrequencySelectiveResidualEnhancer)
            assert block.block_fusion_high_source == "high_raw"
            assert block.gate is None

    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.97).float()

    logits = model(x, return_dict=False)
    assert logits.shape == (2, 1, 64, 64)

    decoder_inputs = {}

    def capture_decoder_high(name):
        def hook(_module, args):
            decoder_inputs[name] = args[2]

        return hook

    dec2_hook = model.dec2.register_forward_pre_hook(capture_decoder_high("dec2"))
    dec1_hook = model.dec1.register_forward_pre_hook(capture_decoder_high("dec1"))
    outputs = model(x, return_features=True)
    dec2_hook.remove()
    dec1_hook.remove()

    assert outputs["logits"].shape == (2, 1, 64, 64)
    for key in ("low0", "low", "residual", "high_raw", "high_hat", "high_for_fusion", "gate"):
        assert key in outputs["features"]
    for high_raw, high_hat, high_for_fusion, gate in zip(
        outputs["features"]["high_raw"],
        outputs["features"]["high_hat"],
        outputs["features"]["high_for_fusion"],
        outputs["features"]["gate"],
    ):
        assert torch.allclose(high_hat, high_raw, atol=1e-6)
        assert torch.allclose(high_for_fusion, high_raw, atol=1e-6)
        assert torch.allclose(gate, torch.ones_like(gate), atol=1e-6)
    assert decoder_inputs["dec2"] is not None
    assert decoder_inputs["dec2"].data_ptr() == outputs["features"]["high_raw"][0].data_ptr()
    assert decoder_inputs["dec1"] is not None
    assert decoder_inputs["dec1"].data_ptr() == outputs["features"]["stage1_residual"][0].data_ptr()

    residual_fusion_cfg = deepcopy(cfg)
    residual_fusion_cfg["model"]["mirfd"]["block_fusion_high_source"] = "residual"
    residual_model = build_model(residual_fusion_cfg)
    residual_outputs = residual_model(x, return_features=True)
    for high_for_fusion, residual, high_raw, high_hat in zip(
        residual_outputs["features"]["high_for_fusion"],
        residual_outputs["features"]["residual"],
        residual_outputs["features"]["high_raw"],
        residual_outputs["features"]["high_hat"],
    ):
        assert torch.allclose(high_for_fusion, residual, atol=1e-6)
        assert torch.allclose(high_hat, high_raw, atol=1e-6)

    ffc_cfg = deepcopy(residual_fusion_cfg)
    ffc_cfg["model"]["mirfd"]["high_enhancer_type"] = "ffc"
    ffc_cfg["model"]["mirfd"]["ffc_gamma_init"] = 0.1
    ffc_cfg["model"]["mirfd"]["ffc_use_highfreq_gate"] = True
    ffc_cfg["model"]["mirfd"]["ffc_highfreq_threshold"] = 0.5
    ffc_cfg["model"]["mirfd"]["ffc_gate_reduction"] = 4
    ffc_model = build_model(ffc_cfg)
    for stage in (ffc_model.stage2, ffc_model.stage3, ffc_model.stage4):
        for block in stage.blocks:
            assert isinstance(block.hfe, FFCFrequencyResidualEnhancer)
            assert block.block_fusion_high_source == "residual"
    ffc_outputs = ffc_model(x, return_features=True)
    assert ffc_outputs["logits"].shape == (2, 1, 64, 64)
    assert ffc_outputs["features"]["high_raw"][0].shape == ffc_outputs["features"]["residual"][0].shape

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
