# MIRFD-Net

PyTorch implementation of Mamba-Induced Residual Frequency Decoupling for infrared small target segmentation.

Experiment logs, current best results, feature-frequency diagnostics, and model-level failure analysis are recorded in [EXPERIMENT_RESULTS_AND_ANALYSIS.md](EXPERIMENT_RESULTS_AND_ANALYSIS.md).

The core idea is to use the Mamba-style 2D propagation output as an adaptive low-frequency semantic approximation, then recover target-sensitive high-frequency details with the input-output residual. In the v2/v2.1 block, the Mamba-induced low representation is lightly calibrated, the raw residual is kept in the high branch, and the gate modulates the high response:

```text
low0      = Align(SS2D(Norm(F)))
low       = LowSmooth(low0)
R         = F - low
high_raw  = HFE(R), Proj(concat(R, HFE(R))), or R + gamma * HFE(R)
G         = TargetAwareGate(low, R)
high_hat  = GateModulation(G, high_raw)
Out       = Fuse(low, high_hat) + F
```

For the v2.1 centered gate:

```text
GateModulation(G, high_raw) = [1 + alpha * (G - 0.5)] * high_raw
```

## MIRFD-Net v2 switches

The v2 implementation keeps SS2D/VMamba-style modeling inside the MIRFD block, not as a full VMamba backbone. It adds the following ablation switches:

- `model.mirfd.use_low_smooth`: applies lightweight low-pass calibration to the Mamba-induced low representation.
- `model.mirfd.high_residual_mode`: `hfe`, `concat_proj`, `add`, or `add_scaled`; `add_scaled` uses `residual + gamma * HFE(residual)`.
- `model.mirfd.high_enhancer_type`: `identity`, `conv_hfe`, or `freq_window`; `freq_window` enables local-window Frequency-Selective Residual Enhancer (FSRE).
- `model.mirfd.gate_mode`: `suppress`, `enhance`, `half_enhance`, or `centered`; `centered` uses `(1 + alpha * (gate - 0.5)) * high_raw`.
- `model.high_skip_stages`: selects which high responses enter decoder skips; currently only stages `{1, 2, 3}` are valid. Stage-4 `high_hat` is exposed for diagnostics and auxiliary heads, but is not a decoder skip unless bottleneck high injection is implemented.
- `model.decoder_high_source`: chooses which MIRFD branch enters decoder high skips: `high_raw`, `high_hat`, or `residual`.
- `model.stage1_high_enhancer_type`: chooses the stage-1 high skip enhancer independently: `identity`, `conv_hfe`, or `freq_window`.
- `model.use_stage1_high_skip`: legacy switch for adding shallow high-frequency skip information when `high_skip_stages` is not set.
- `model.use_aux_heads`: when enabled, auxiliary heads still supervise b2/b3/b4 `high_hat`; this is separate from whether a stage is used as a decoder high skip.
- `loss.spectral_high_target`: chooses `residual`, `high_raw`, or `high_hat` for high-branch spectral regularization.
- `loss.gate_aux_weight` / `loss.gate_bg_weight`: optional light supervision for gate target-awareness.

Ready-to-run v2 configs:

```text
configs/mirfd_nuaa_sirst_ss2d_v2.yaml
configs/mirfd_nudt_sirst_ss2d_v2.yaml
configs/mirfd_irstd_1k_ss2d_v2.yaml
```

## What is included

- `mirfd.models.MIRFDNet`
- `mirfd.models.MIRFDBlock`
- `HighFrequencyEnhancer` and `TargetAwareGate`
- self-contained `Mamba2D` fallback with four-direction selective scans
- optional VMamba-style `SS2D` branch using `mamba_ssm` selective scan
- optional external VMamba/SS2D adapter and fallback+SS2D parallel branch
- BCE + Dice + auxiliary high-frequency loss + soft spectral regularization
- flexible dataset loader for NUDT-SIRST / IRSTD-1K / NUAA-SIRST style folders
- train, test, inference, and FFT visualization scripts
- ablation switches: `mamba_residual`, `avgpool`, `laplace`, `sobel`, gate on/off, fusion type
- metrics: IoU, nIoU, Dice, precision, recall, Pd, Fa

## Quick smoke test

```bash
python tests/smoke_test.py
```

## Train

Edit `configs/mirfd_default.yaml`, especially `data.root`, then run:

```bash
python scripts/train.py --config configs/mirfd_default.yaml --output-dir runs/mirfd
```

You can also override the dataset root:

```bash
python scripts/train.py --config configs/mirfd_default.yaml --data-root /path/to/NUDT-SIRST
```

The dataset loader first tries common layouts such as:

```text
root/train/images + root/train/masks
root/images/train + root/masks/train
root/images       + root/masks
```

If your layout is different, set `train_image_dir`, `train_mask_dir`, `val_image_dir`, `val_mask_dir`, `test_image_dir`, and `test_mask_dir` in the config.

## Evaluate

```bash
python scripts/test.py --config configs/mirfd_default.yaml --checkpoint runs/mirfd/best.pt --split test
```

`Pd` is target-level detection probability: a ground-truth connected component is counted as detected when the predicted mask overlaps it. `Fa` is pixel-level false alarm rate: false-positive background pixels divided by all pixels.

## Inference

```bash
python scripts/infer.py --config configs/mirfd_default.yaml --checkpoint runs/mirfd/best.pt --input /path/to/image_or_dir --output-dir outputs/infer
```

## FFT branch visualization

```bash
python scripts/visualize_fft.py --config configs/mirfd_default.yaml --checkpoint runs/mirfd/best.pt --image /path/to/image.png
```

The saved panel is: input, low-branch spectrum, high-branch spectrum, gate map.

## Replace the fallback Mamba block

`mirfd/models/mamba2d.py` is the self-contained fallback. For a VMamba-style branch, use:

```yaml
model:
  mamba:
    variant: ss2d
    scan_backend: auto
```

`scan_backend: auto` tries `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` first and falls back to the built-in PyTorch reference selective scan when the CUDA extension is unavailable. Use `scan_backend: cuda` to require `mamba_ssm`, or `scan_backend: ref` to force the reference path. The ready-to-edit config is `configs/mirfd_ss2d.yaml`.

To run the fallback and SS2D branches in parallel:

```yaml
model:
  mamba:
    variant: parallel
    parallel_real_variant: ss2d
    parallel_fusion: concat
```

The ready-to-edit config is `configs/mirfd_parallel_ss2d.yaml`.

To adapt an external VMamba block:

```yaml
model:
  mamba:
    variant: external
    external_import_path: your_package.your_module.SS2D
    external_layout: auto
    external_kwargs: {}
```

The external block must accept the channel dimension as its first constructor argument and return either BCHW or BHWC output with the same shape semantics.
