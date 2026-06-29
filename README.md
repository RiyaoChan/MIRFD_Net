# MIRFD-Net

PyTorch implementation of Mamba-Induced Residual Frequency Decoupling for infrared small target segmentation.

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
- BCE + Dice + auxiliary high-frequency loss + soft spectral regularization
- flexible dataset loader for NUDT-SIRST / IRSTD-1K / NUAA-SIRST style folders
- train, test, inference, and FFT visualization scripts
- ablation switches: `mamba_residual`, `avgpool`, `laplace`, `sobel`, gate on/off, fusion type

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

`mirfd/models/mamba2d.py` is intentionally isolated. To use a full VMamba/SS2D implementation, keep the same BCHW input/output contract and replace `Mamba2D` or pass a compatible block into `MIRFDBlock`.
