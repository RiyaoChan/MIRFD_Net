# MIRFD-Net

PyTorch implementation of Mamba-Induced Residual Frequency Decoupling for infrared small target segmentation.

Experiment logs, current best results, feature-frequency diagnostics, and model-level failure analysis are recorded in [EXPERIMENT_RESULTS_AND_ANALYSIS.md](EXPERIMENT_RESULTS_AND_ANALYSIS.md).

The core idea is to use the Mamba-style 2D propagation output as an adaptive low-frequency semantic approximation, then recover target-sensitive high-frequency details with the input-output residual:

```text
F_l = Align(Mamba2D(Norm(F)))
R   = F - F_l
F_h = HighFrequencyEnhancer(R)
G   = TargetAwareGate(F_l, R)
Out = Fuse(F_l, G * F_h) + F
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
