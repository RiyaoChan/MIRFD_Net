# MIRFD-Net 实验结果与模型分析记录

记录时间：2026-06-29  
最近更新：2026-07-01
服务器路径：`/DATA20T/bip/cry/code/MIRFD_Net`  
数据集根目录：`/DATA20T/bip/cry/code/SIRST-5K-main/dataset/`

本文记录当前 MIRFD-Net 在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上的训练结果、训练策略变化、模型分支诊断，以及后续改进方向。

## 1. 当前最佳结果

三组最有效的配置都采用 SCTransNet-style 数据预处理，包括 raw intensity mean/std 标准化、`256x256` 正样本优先 crop、翻转/转置增强、AdamW、cosine warmup，以及 centroid Pd/Fa 统计。

| Dataset | Best run | Config | Best epoch | IoU | nIoU | Dice | Precision | Recall | Pd | Fa | Compared with previous SS2D |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `nuaa_sirst_ss2d_sctrans_adamw_lr1e3` | `configs/mirfd_nuaa_sirst_ss2d_sctrans_adamw_lr1e3.yaml` | 374 | 0.7452 | 0.7184 | 0.8540 | 0.8443 | 0.8639 | 0.9696 | 0.000017 | +0.0320 IoU |
| NUDT-SIRST | `v2_2_ablation/nudt_stage1_identity_stage2_fsre` | `configs/mirfd_nudt_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml` | 500 | 0.8756 | 0.8959 | 0.9337 | 0.9348 | 0.9325 | 0.9841 | 0.000010 | +0.0778 IoU |
| IRSTD-1K | `v2_1_ablation/irstd_shallow_high_skip` | `configs/mirfd_irstd_1k_ss2d_v2_1_shallow_high_skip.yaml` | 340 | 0.6290 | 0.5392 | 0.7723 | 0.7137 | 0.8413 | 0.8469 | 0.000011 | +0.0265 IoU |

对应 checkpoint：

| Dataset | Best checkpoint |
|---|---|
| NUAA-SIRST | `runs/nuaa_sirst_ss2d_sctrans_adamw_lr1e3/best.pt` |
| NUDT-SIRST | `runs/v2_2_ablation/nudt_stage1_identity_stage2_fsre/best.pt` |
| IRSTD-1K | `runs/v2_1_ablation/irstd_shallow_high_skip/best.pt` |

## 2. 与原始实验对比

| Dataset | Run | Epochs | Best epoch | Best IoU | nIoU | Dice | Pd | Fa | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | `nuaa_sirst_ss2d` | 300 | 166 | 0.7132 | 0.7070 | 0.8326 | 0.9848 | 0.000100 | 原 SS2D 策略 |
| NUAA-SIRST | `nuaa_sirst_fallback` | 300 | 201 | 0.6975 | 0.6954 | 0.8218 | 0.9658 | 0.000100 | fallback Mamba |
| NUAA-SIRST | `nuaa_sirst_ss2d_sctrans_adamw_lr3e3` | 500 | 212 | 0.7320 | 0.7200 | 0.8453 | 0.9620 | 0.000021 | 新预处理，lr=0.003 |
| NUAA-SIRST | `nuaa_sirst_ss2d_sctrans_adamw_lr1e3` | 500 | 374 | 0.7452 | 0.7184 | 0.8540 | 0.9696 | 0.000017 | 当前最佳 |
| NUDT-SIRST | `nudt_sirst_ss2d` | 300 | 281 | 0.7978 | 0.8173 | 0.8875 | 0.9746 | 0.000100 | 原 SS2D 策略 |
| NUDT-SIRST | `nudt_sirst_fallback` | 300 | 255 | 0.7847 | 0.8040 | 0.8793 | 0.9788 | 0.000100 | fallback Mamba |
| NUDT-SIRST | `nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3` | 500 | 441 | 0.8696 | 0.8926 | 0.9302 | 0.9799 | 0.000011 | 当前最佳 |
| NUDT-SIRST | `nudt_sirst_ss2d_sctrans_adamw_bs32_lr3e3` | 500 | 491 | 0.8473 | 0.8666 | 0.9174 | 0.9778 | 0.000013 | lr=0.003，不如 lr=0.001 |
| IRSTD-1K | `irstd_1k_ss2d` | 300 | 91 | 0.5939 | 0.5717 | 0.7452 | 0.8946 | 0.000100 | 原 SS2D 策略 |
| IRSTD-1K | `irstd_1k_fallback` | 300 | 198 | 0.5916 | 0.5606 | 0.7434 | 0.9150 | 0.000100 | fallback Mamba |
| IRSTD-1K | `irstd_1k_ss2d_sctrans_adamw_bs32_lr1e3` | 500 | 398 | 0.6025 | 0.5269 | 0.7519 | 0.8605 | 0.000026 | 旧最佳，但提升有限 |
| IRSTD-1K | `irstd_1k_ss2d_sctrans_adamw_bs32_lr3e3` | 500 | 63 | 0.5437 | 0.4645 | 0.7044 | 0.7891 | 0.000044 | epoch 67 后 NaN |

## 2.1 MIRFD-Net v2 spectral ablation 结果（2026-06-30）

本轮实验固定 v2 结构，只对比 `spectral_low_weight/spectral_high_weight=0.001` 与关闭 spectral loss，目的是区分收益来自结构还是来自频谱约束。三组数据集均单独训练 500 epoch，输出目录为 `runs/v2_ablation/`。

| Dataset | Run | Spectral | Best IoU epoch | IoU | nIoU | Dice | Best Pd-Fa | Compared with previous best IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `v2_ablation/nuaa_v2_spectral` | yes | 255 | 0.7209 | 0.7116 | 0.8378 | 0.9696 | -0.0243 |
| NUAA-SIRST | `v2_ablation/nuaa_v2_no_spectral` | no | 220 | 0.7150 | 0.7041 | 0.8338 | 0.9733 | -0.0302 |
| NUDT-SIRST | `v2_ablation/nudt_v2_spectral` | yes | 368 | 0.8349 | 0.8557 | 0.9100 | 0.9873 | -0.0347 |
| NUDT-SIRST | `v2_ablation/nudt_v2_no_spectral` | no | 462 | 0.8607 | 0.8800 | 0.9251 | 0.9820 | -0.0089 |
| IRSTD-1K | `v2_ablation/irstd_v2_spectral` | yes | 351 | 0.6029 | 0.5494 | 0.7523 | 0.9147 | +0.0004 |
| IRSTD-1K | `v2_ablation/irstd_v2_no_spectral` | no | 160 | 0.6129 | 0.5311 | 0.7600 | 0.9217 | +0.0104 |

表中 IoU/nIoU/Dice 是 best-IoU checkpoint 同一 epoch 的指标；`Best Pd-Fa` 为该 run 内 `pd - fa` 的最佳值，可能来自不同 epoch。

对比结论：

1. 相比上一轮 SCTransNet-style 最佳结果，v2 在 NUAA-SIRST 和 NUDT-SIRST 上没有提升；NUAA 最优从 `0.7452` 降到 `0.7209`，NUDT 最优从 `0.8696` 降到 `0.8607`。
2. IRSTD-1K 有提升，最佳从 `0.6025` 提升到 `0.6129`，提升来自 v2 结构本身的 no-spectral 版本。
3. spectral loss 不是稳定正收益：NUAA spectral 比 no-spectral 高 `+0.0059` IoU，但 NUDT 低 `-0.0258`，IRSTD-1K 低 `-0.0100`。
4. 当前证据支持“v2 结构对 IRSTD 更有帮助，但 spectral 约束会在 NUDT/IRSTD 上损害分割 IoU/Dice”。后续主实验应优先保留 v2 结构、默认关闭 spectral loss，只把 spectral 作为诊断或弱权重消融。

## 3. 训练异常记录

以下实验出现 NaN，不建议使用其 `last.pt`，只可参考 NaN 前保存的 `best.pt`：

| Run | First NaN epoch | Comment |
|---|---:|---|
| `irstd_1k_ss2d_sctrans_adamw_bs32_lr3e3` | 67 | lr=0.003 在 IRSTD-1K 上不稳定 |
| `nuaa_sirst_ss2d_sctrans_adamw_bs8` | 64 | batch size 8 + lr=0.003 不稳定，且效果不如 lr=0.001 |
| `sirst5k_three_fallback` | 45 | 混合数据集旧实验，不作为主要结论 |
| `v2_1_ablation/irstd_centered_shallow_add_scaled` | 249 | `add_scaled` 在 IRSTD-1K 上后期出现 non-finite loss；可参考 NaN 前 `best.pt`，但不建议作为主配置 |

当前训练脚本没有在 loss 出现 NaN 时自动中止，因此 NaN 后仍会继续保存 `last.pt`。后续应在 `scripts/train.py` 中加入有限值检查：

```python
if not torch.isfinite(loss):
    raise FloatingPointError(f"non-finite loss at epoch={epoch}")
```

## 4. 当前模型结构摘要

当前主模型不是直接使用完整 VMamba 作为 backbone，而是在 MIRFD Block 内部将 SS2D/VMamba-style block 作为 Mamba 分支：

```text
F_norm = Norm(F)
F_m    = SS2D(F_norm)
F_l    = Align(F_m)
R      = F - F_l
F_h    = HighFrequencyEnhancer(R)
G      = TargetAwareGate(F_l, R)
Out    = Fuse(F_l, G * F_h) + F
```

主干下采样关系：

| Stage | Resolution for 256x256 input | Module |
|---|---:|---|
| Stem + Stage 1 | 128x128 | ConvStage |
| Stage 2 | 64x64 | MIRFDStage |
| Stage 3 | 32x32 | MIRFDStage |
| Stage 4 | 16x16 | MIRFDStage |
| Decoder | 32 -> 64 -> 128 -> 256 | U-Net style decoder |

对于红外小目标，`16x16` 的深层特征已经非常粗，stage3/stage4 难以恢复细粒度边界。

## 5. 分支频谱与响应诊断

诊断方法：加载每个数据集当前最佳 checkpoint，在验证集前 64 张图上统计：

- `low_lowratio`：low 分支低频能量占比。
- `high_lowratio`：high 分支低频能量占比，越低越偏高频。
- `high/low abs`：high 分支平均幅值相对 low 分支的比例。
- `gate fg-bg`：目标区域 gate 均值减背景区域 gate 均值，正值表示目标区域被更强增强。

### 5.1 图像、GT 与预测频谱

| Dataset | Image low ratio | Mask low ratio | Prediction low ratio | Pred fg mean | Pred bg mean |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | 0.5341 | 0.2668 | 0.2777 | 0.8746 | 0.000117 |
| NUDT-SIRST | 0.5604 | 0.2294 | 0.2320 | 0.9046 | 0.000049 |
| IRSTD-1K | 0.5041 | 0.1613 | 0.2037 | 0.8282 | 0.000107 |

预测图频谱接近 GT mask，比输入图更偏高频集中，说明分割输出本身没有明显过度平滑。

### 5.2 MIRFD 分支频谱

| Dataset | Stage | low lowratio | high lowratio | residual lowratio | high/low abs | residual/low abs |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 2 | 0.1592 | 0.1249 | 0.1476 | 0.8140 | 1.0596 |
| NUAA-SIRST | 3 | 0.3654 | 0.2310 | 0.2230 | 0.6901 | 1.3564 |
| NUAA-SIRST | 4 | 0.1783 | 0.1470 | 0.1730 | 0.5404 | 1.4736 |
| NUDT-SIRST | 2 | 0.1349 | 0.1229 | 0.1319 | 0.9352 | 1.1514 |
| NUDT-SIRST | 3 | 0.2434 | 0.2138 | 0.1909 | 0.6454 | 1.4801 |
| NUDT-SIRST | 4 | 0.1470 | 0.1343 | 0.1786 | 0.3852 | 1.2861 |
| IRSTD-1K | 2 | 0.3350 | 0.1992 | 0.2043 | 1.1909 | 1.0975 |
| IRSTD-1K | 3 | 0.3475 | 0.2262 | 0.3133 | 0.6782 | 1.6683 |
| IRSTD-1K | 4 | 0.2718 | 0.1560 | 0.2430 | 0.4577 | 1.4229 |

观察：

1. `low` 和 `high` 的频谱差距存在，但并不强。当前代码只是把 Mamba 输出命名为 `low`，没有显式低通约束。
2. stage4 的 `high/low abs` 明显偏低，说明深层高频补偿弱化。
3. IRSTD-1K 的 stage2/stage3 low 分支低频占比更高，但最终效果仍差，说明问题不只是低频不足，还包括目标级细粒度建模和 gate 稳定性。

### 5.3 目标区域响应与 gate

| Dataset | Stage | high fg/bg | low fg/bg | residual fg/bg | gate mean | gate fg-bg |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 2 | 4.9645 | 1.9167 | 3.3594 | 0.6787 | 0.0323 |
| NUAA-SIRST | 3 | 3.7318 | 0.9354 | 6.2055 | 0.6911 | -0.0461 |
| NUAA-SIRST | 4 | 2.8799 | 0.9514 | 2.4674 | 0.6813 | 0.0593 |
| NUDT-SIRST | 2 | 2.6954 | 3.0342 | 3.5313 | 0.7104 | -0.1586 |
| NUDT-SIRST | 3 | 3.6911 | 0.7608 | 2.8750 | 0.5957 | 0.0685 |
| NUDT-SIRST | 4 | 5.9602 | 1.1242 | 2.1914 | 0.5680 | -0.1020 |
| IRSTD-1K | 2 | 3.1559 | 0.9412 | 3.2283 | 0.6382 | 0.1107 |
| IRSTD-1K | 3 | 4.0747 | 1.0978 | 2.0070 | 0.6899 | -0.1608 |
| IRSTD-1K | 4 | 6.2088 | 1.0096 | 2.1406 | 0.7627 | -0.2031 |

观察：

1. high 分支确实更关注目标区域，`high fg/bg` 大多明显高于 1。
2. gate 并不稳定。多个 stage 的 `gate fg-bg` 为负，表示目标区域 gate 比背景区域更低，可能抑制小目标高频响应。
3. IRSTD-1K 的 stage3/stage4 gate 抑制最明显，这可能解释其 IoU 提升很小。

## 6. 当前 SOTA 差距的模型层面判断

当前结果距离 SOTA 仍有差距，主要瓶颈不是单纯训练策略，而是模型结构对“低频语义近似 + 高频细节恢复”的约束还不够强。

### 6.1 Mamba low 分支不够纯

当前实现：

```python
fm = self.mamba(self.norm(x))
low = self.align(fm)
residual = x - low
```

问题是 `low` 没有显式 low-pass 操作，只靠 Mamba 的低频偏好和很弱的 spectral loss。诊断结果显示 low/high 分支频谱分离不够强。

### 6.2 深层 high 分支偏弱

stage4 的 high 幅值相对 low 明显偏低：

| Dataset | Stage4 high/low abs |
|---|---:|
| NUAA-SIRST | 0.5404 |
| NUDT-SIRST | 0.3852 |
| IRSTD-1K | 0.4577 |

这说明越深层越依赖低频语义，局部高频补偿不够。对于小目标，stage4 的 `16x16` 特征已经过粗。

### 6.3 浅层高频没有充分利用

decoder 最后一层当前写法是：

```python
d1 = self.dec1(d2, e1, torch.zeros_like(e1))
```

stage1 的高分辨率浅层特征没有显式 high residual skip。红外小目标最关键的边缘、亮点和局部突变往往在浅层，当前结构没有充分建模。

### 6.4 gate 可能误抑制目标

当前：

```python
high_hat = gate * high
```

如果 gate 在目标区域偏低，就会直接压制目标高频。诊断显示 NUDT 和 IRSTD 的部分 stage 确实存在这种情况。

## 7. 下一步优先改进方向

建议优先做 MIRFD Block v2，而不是继续只调 batch size 或学习率。

### 7.1 显式低通 Mamba low

将：

```python
low = self.align(fm)
```

改成：

```python
low = self.low_smooth(self.align(fm))
```

候选实现：

- depthwise blur conv。
- avgpool low-pass。
- learnable low-pass kernel。

目标是让 `F_l` 更稳定地成为低频背景/语义近似。

### 7.2 高频分支保留 residual 直连

当前：

```python
high = self.hfe(residual)
```

建议改成：

```python
high = residual + self.hfe(residual)
```

或：

```python
high = self.high_proj(torch.cat([residual, self.hfe(residual)], dim=1))
```

避免 HFE 把原始高频残差信息洗掉。

### 7.3 gate 改成不抑制型

当前：

```python
high_hat = gate * high
```

建议改成：

```python
high_hat = (0.5 + gate) * high
```

或：

```python
high_hat = (1.0 + gate) * high
```

这样 gate 更像增强权重，而不是硬性压制开关。

### 7.4 加入 stage1 浅层高频 skip

给 stage1 加轻量高频分支，例如 Laplace/HFE：

```text
e1_high = HFE(e1 - AvgPool(e1))
d1 = dec1(d2, e1, e1_high)
```

这能直接补偿小目标浅层边缘和亮点。

### 7.5 IRSTD-1K 单独策略

IRSTD-1K 当前提升有限，建议单独做：

- `lr=0.0005` 或 `lr=0.0008`。
- `batch_size=16`，提高小目标样本梯度随机性。
- 减弱或关闭 spectral loss 做 ablation。
- gate v2 与 stage1 high skip 优先在 IRSTD-1K 上验证。

## 8. 后续实验建议

| Priority | Experiment | Purpose |
|---:|---|---|
| P0 | MIRFD v2: explicit low-pass + residual high direct path | 验证低/高频分工是否更清晰 |
| P0 | non-suppressive gate | 验证 gate 是否误抑制小目标 |
| P1 | stage1 shallow high skip | 强化浅层细粒度目标响应 |
| P1 | IRSTD bs16/lr5e-4 | 提升 IRSTD 稳定性 |
| P2 | spectral loss ablation | 判断软频谱约束是否带来真实收益 |
| P2 | residual type ablation: avgpool/laplace/sobel | 证明 Mamba-induced residual 相比固定高频算子是否有优势 |

## 9. 当前结论

SCTransNet-style 训练策略显著改善了 NUAA-SIRST 和 NUDT-SIRST，但 IRSTD-1K 仍然提升有限。模型分析显示，当前 MIRFD-Net 的核心 idea 已经有一定效果，但实现上还没有形成足够强的低/高频解耦：

- Mamba low 分支没有显式低通，低频语义近似不够纯。
- high 分支在深层偏弱，浅层高频没有充分利用。
- gate 在部分数据集和 stage 上会抑制目标区域。
- SS2D 主要位于 1/4、1/8、1/16 特征，对极小目标细粒度响应不足。

下一步应优先从模型结构改进，而不是继续只调训练超参。

## 10. MIRFD-Net v2 代码实现记录

根据 `MIRFD_Net_v2_improvement_plan_for_codex.md`，当前代码已完成 MIRFD-Net v2 的主体实现，并保留 v1 兼容配置。

### 10.1 已采用的方案

| Module | Adopted scheme | Config switch |
|---|---|---|
| Low branch | 对 Mamba/SS2D 输出后的 `low0` 做 `LowSmooth` 低通校准，而不是对输入特征预先手工分频 | `model.mirfd.use_low_smooth`, `model.mirfd.low_smooth_beta_init` |
| High branch | 采用推荐方案 A：`high_raw = Conv1x1(concat(residual, HFE(residual)))`，保留 residual 直连 | `model.mirfd.high_residual_mode: concat_proj` |
| Gate | 采用增强型 gate：`high_hat = (1 + alpha * gate) * high_raw`，避免旧式 `gate * high` 直接压制目标 | `model.mirfd.gate_mode: enhance`, `model.mirfd.gate_alpha_init` |
| Decoder skip | decoder 使用 gate 后的 `high_hat`，保持目标相关高频增强进入解码器 | internal |
| Stage1 high skip | 增加浅层 high skip：`e1_high = HFE(e1 - Blur(e1))`，替代原来的 `zeros_like(e1)` | `model.use_stage1_high_skip` |
| Spectral loss | high spectral loss 可选 `residual/high_raw/high_hat`；v2 配置默认使用 `high_raw` | `loss.spectral_high_target` |
| Training stability | 非有限 loss 直接中止；额外保存 `last_finite.pt` 和多指标 best checkpoint | `train.clip_grad_norm` / `train.grad_clip_norm` |
| Pyramid avgpool ablation | 增加轻量 pyramid residual 选项，用 avgpool+nearest upsample 近似低频并取残差；这不是严格 Haar wavelet | `model.mirfd.residual_type: pyramid_avgpool` |

### 10.2 新增/修改文件

| File | Change |
|---|---|
| `mirfd/models/mirfd_block.py` | 新增 `FixedDepthwiseBlur`、`LowSmooth`、`high_residual_mode`、`gate_mode`、`pyramid_avgpool` residual，并返回 `low0/low/residual/high_raw/high_hat/gate` |
| `mirfd/models/mirfd_net.py` | 接入 Stage1 high skip，decoder 使用 `high_hat`，feature dict 暴露 v2 诊断特征 |
| `mirfd/losses.py` | 增加 `spectral_high_target`，支持对 `high_raw` 或 `residual` 做高频正则 |
| `scripts/train.py` | 增加 NaN 保护、`grad_clip_norm` 别名、多指标 checkpoint |
| `scripts/visualize_fft.py` | 可视化 `low/residual/high_raw/high_hat/gate` 及频谱 |
| `configs/mirfd_nuaa_sirst_ss2d_v2.yaml` | NUAA v2 配置，`lr=0.001`，spectral weight 0.001 |
| `configs/mirfd_nudt_sirst_ss2d_v2.yaml` | NUDT v2 配置，`lr=0.001`，`batch_size=32` |
| `configs/mirfd_irstd_1k_ss2d_v2.yaml` | IRSTD v2 配置，`lr=0.0005`，`batch_size=16`，先关闭 spectral loss |

### 10.3 默认兼容性

旧配置的模型结构默认仍保持 v1 行为：

```yaml
model:
  mirfd:
    use_low_smooth: false
    high_residual_mode: hfe
    gate_mode: suppress
  use_stage1_high_skip: false
loss:
  spectral_high_target: high_raw
```

旧实验 checkpoint 对应的结构语义仍然清晰。需要注意的是，`MIRFDLoss` 的默认高频频谱约束目标已改为 `high_raw`；如果要严格复现实验前的 gate 后高频约束，可在旧 config 中显式设置 `loss.spectral_high_target: high` 或 `high_hat`。v2 实验建议继续使用新增 v2 配置中的 `high_raw`。

### 10.4 下一步建议实验

优先启动以下三组：

```bash
python scripts/train.py --config configs/mirfd_nuaa_sirst_ss2d_v2.yaml --output-dir runs/nuaa_sirst_ss2d_v2
python scripts/train.py --config configs/mirfd_nudt_sirst_ss2d_v2.yaml --output-dir runs/nudt_sirst_ss2d_v2
python scripts/train.py --config configs/mirfd_irstd_1k_ss2d_v2.yaml --output-dir runs/irstd_1k_ss2d_v2
```

IRSTD-1K 建议先观察前 80 epoch 是否稳定。如果稳定但 IoU 不够，再单独加 `spectral_low_weight=0.001`、`spectral_high_weight=0.001` 做对照。

## 11. MIRFD-Net v2.1 centered gate 实验计划（2026-06-30）

根据 `MIRFD_Net_v2_1_centered_gate_improvement_plan.md` 和 `docs/visualizations/v2_feature_diagnostics`，v2 的主要问题不是 high branch 没有高频响应，而是 `high_hat` 经 enhance gate 后变得过强、过宽泛，背景纹理和深层 coarse residual 也被送入 decoder。因此 v2.1 优先验证三个结构改动：

| Change | Motivation | Config switch |
|---|---|---|
| Centered gate | 让 gate 从只增强 `[1, 2]` 的 scale 变成可增强/可抑制 `[0.5, 1.5]` 的选择器 | `model.mirfd.gate_mode: centered` |
| Selective high skip | 只保留浅层和 stage2 decoder high skip，关闭 stage-3 decoder high skip；stage-4 high_hat 当前不作为 decoder skip 使用 | `model.high_skip_stages: [1, 2]` |
| add_scaled HFE | 保留 `residual = F - low` 原义，限制 HFE 对 residual 的过度重构 | `model.mirfd.high_residual_mode: add_scaled`, `hfe_scale_init: 0.1` |

新增代码开关：

```text
high_hat = (1 + alpha * (gate - 0.5)) * high_raw       # centered gate
high_raw = residual + gamma * HFE(residual)            # add_scaled
high_skip_stages = [1, 2]                              # shallow high skips
```

第一轮 v2.1 消融固定训练策略为上一轮 v2 no-spectral：SCTransNet-style 数据增强、AdamW、cosine warmup、关闭 spectral loss。这样对比尽量只反映结构变化。

| Experiment | Purpose | Gate | High skip stages | High residual |
|---|---|---|---|---|
| A `centered_gate` | 只验证 centered gate 是否缓解 high_hat 过强 | centered | `[1,2,3]` | concat_proj |
| B `shallow_high_skip` | 只验证关闭 stage-3 decoder high skip 是否减少背景污染；stage-4 high_hat 当前不作为 decoder skip 使用 | enhance | `[1,2]` | concat_proj |
| C `centered_shallow` | 同时验证 centered gate + shallow high skip | centered | `[1,2]` | concat_proj |
| D `centered_shallow_add_scaled` | 在 C 基础上限制 HFE 重构强度 | centered | `[1,2]` | add_scaled |

新增配置文件覆盖 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三个数据集：

```text
configs/mirfd_<dataset>_ss2d_v2_1_centered_gate.yaml
configs/mirfd_<dataset>_ss2d_v2_1_shallow_high_skip.yaml
configs/mirfd_<dataset>_ss2d_v2_1_centered_shallow.yaml
configs/mirfd_<dataset>_ss2d_v2_1_centered_shallow_add_scaled.yaml
```

可选 gate loss 已实现但第一轮不启用：

```yaml
loss:
  gate_aux_weight: 0.0
  gate_bg_weight: 0.0
```

如果 C/D 的可视化仍显示 gate 对背景大面积发亮，再开启 `gate_bg_weight: 0.01` 做第二轮。

### 11.1 v2.1 实验结果（2026-06-30）

12 组 v2.1 消融已经在服务器跑完，输出目录为 `runs/v2_1_ablation/`。除 `irstd_centered_shallow_add_scaled` 在 epoch 249 出现 NaN 外，其余实验均完成 500 epoch；失败组仍保留 NaN 前的 `best.pt`，best-IoU epoch 为 235。

| Dataset | Run | Best epoch | IoU | nIoU | Dice | Precision | Recall | Pd | Fa | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | `nuaa_centered_gate` | 367 | 0.7028 | 0.6985 | 0.8254 | 0.8308 | 0.8201 | 0.9544 | 0.000032 | 完成 |
| NUAA-SIRST | `nuaa_shallow_high_skip` | 259 | 0.7267 | 0.7127 | 0.8417 | 0.8394 | 0.8441 | 0.9582 | 0.000024 | 完成 |
| NUAA-SIRST | `nuaa_centered_shallow` | 196 | 0.6966 | 0.6910 | 0.8212 | 0.8291 | 0.8134 | 0.9430 | 0.000029 | 完成 |
| NUAA-SIRST | `nuaa_centered_shallow_add_scaled` | 182 | 0.7211 | 0.7048 | 0.8379 | 0.8328 | 0.8431 | 0.9658 | 0.000026 | 完成 |
| NUDT-SIRST | `nudt_centered_gate` | 451 | 0.8341 | 0.8542 | 0.9095 | 0.9126 | 0.9065 | 0.9757 | 0.000015 | 完成 |
| NUDT-SIRST | `nudt_shallow_high_skip` | 378 | 0.8618 | 0.8826 | 0.9257 | 0.9279 | 0.9236 | 0.9788 | 0.000013 | 完成 |
| NUDT-SIRST | `nudt_centered_shallow` | 479 | 0.8539 | 0.8700 | 0.9212 | 0.9322 | 0.9105 | 0.9788 | 0.000008 | 完成 |
| NUDT-SIRST | `nudt_centered_shallow_add_scaled` | 363 | 0.8156 | 0.8328 | 0.8984 | 0.9002 | 0.8967 | 0.9746 | 0.000011 | 完成 |
| IRSTD-1K | `irstd_centered_gate` | 300 | 0.6091 | 0.5245 | 0.7570 | 0.6907 | 0.8375 | 0.8367 | 0.000014 | 完成 |
| IRSTD-1K | `irstd_shallow_high_skip` | 340 | 0.6290 | 0.5392 | 0.7723 | 0.7137 | 0.8413 | 0.8469 | 0.000011 | 完成 |
| IRSTD-1K | `irstd_centered_shallow` | 374 | 0.5972 | 0.5442 | 0.7478 | 0.6642 | 0.8554 | 0.8707 | 0.000050 | 完成 |
| IRSTD-1K | `irstd_centered_shallow_add_scaled` | 235 | 0.5947 | 0.5081 | 0.7458 | 0.6688 | 0.8430 | 0.8435 | 0.000031 | epoch 249 NaN |

与上一轮 v2 no-spectral 对比：

| Dataset | v2 no-spectral IoU | Best v2.1 run | Best v2.1 IoU | Change |
|---|---:|---|---:|---:|
| NUAA-SIRST | 0.7150 | `nuaa_shallow_high_skip` | 0.7267 | +0.0117 |
| NUDT-SIRST | 0.8607 | `nudt_shallow_high_skip` | 0.8618 | +0.0011 |
| IRSTD-1K | 0.6129 | `irstd_shallow_high_skip` | 0.6290 | +0.0161 |

与当前全局最佳对比：

| Dataset | Current global best before v2.1 | Best v2.1 | Conclusion |
|---|---:|---:|---|
| NUAA-SIRST | 0.7452 | 0.7267 | v2.1 仍低 0.0185，暂不替换 NUAA 主配置 |
| NUDT-SIRST | 0.8696 | 0.8618 | v2.1 仍低 0.0078，暂不替换 NUDT 主配置 |
| IRSTD-1K | 0.6129 | 0.6290 | v2.1 提升 0.0161，刷新 IRSTD 当前最佳 |

结论：

1. `shallow_high_skip` 是这一轮唯一稳定有效的结构改动，三个数据集相对 v2 no-spectral 均不下降，并在 IRSTD-1K 上带来明确提升。
2. `centered_gate` 单独使用不稳定，NUAA、NUDT、IRSTD 均低于对应的 `shallow_high_skip`，说明 centered gate 没有直接解决 high response 背景扩散问题。
3. `centered + shallow` 仍弱于 shallow-only，说明当前 gate 的调制方式可能仍会干扰目标高频细节，不建议作为默认主配置。
4. `add_scaled` 没有带来稳定收益，且 IRSTD-1K 出现 NaN，暂不作为主实验路线。
5. 后续主配置建议保留：
   - `configs/mirfd_nuaa_sirst_ss2d_v2_1_shallow_high_skip.yaml`
   - `configs/mirfd_nudt_sirst_ss2d_v2_1_shallow_high_skip.yaml`
   - `configs/mirfd_irstd_1k_ss2d_v2_1_shallow_high_skip.yaml`

当前最佳结果表已同步更新：IRSTD-1K 的全局最佳从 `v2_ablation/irstd_v2_no_spectral` 替换为 `v2_1_ablation/irstd_shallow_high_skip`；NUAA-SIRST 和 NUDT-SIRST 仍沿用 SCTransNet-style AdamW 最佳配置。

### 11.2 内部特征统计脚本

新增 `scripts/diagnose_feature_statistics.py`，建议将输出保存到 `docs/diagnostics/feature_statistics/`。该 CSV 用于量化 MIRFD 内部分支 `low/residual/high_raw/gate/high_hat` 的频谱属性、目标选择性和 false alarm 关系，重点验证 residual 是否更高频、HFE/gate 是否提升目标相关性，以及深层 high response / decoder skip 是否更容易引入背景残差。脚本同时统计 stage-1 的 `stage1_low/stage1_residual/stage1_high`；stage-1 没有 gate，因此 gate 相关字段写为 `nan`。`stage_enabled` 表示该 stage 的 high 分支是否有效，`stage_used_as_decoder_skip` 表示该 stage 是否真的进入 decoder，其中 stage-4 当前固定为 0。需要注意：`pred_iou` 和 `pred_has_false_alarm` 是样本级最终预测指标，会重复写入同一样本的各 stage 行，不是 stage 自身的输出指标。若 `use_aux_heads=True`，auxiliary heads 仍会监督 b2/b3/b4 的 `high_hat`，这与 decoder high skip 是否启用是两件事。

### 11.3 v2.1 shallow high skip 内部特征统计（2026-06-30）

已对三个数据集的 `v2_1_ablation/*_shallow_high_skip/best.pt` 生成测试集内部特征统计：

| Dataset | Raw CSV | Summary CSV | Samples |
|---|---|---|---:|
| NUAA-SIRST | `docs/diagnostics/feature_statistics/nuaa_v2_1_shallow_high_skip.csv` | `docs/diagnostics/feature_statistics/summary_nuaa_v2_1_shallow_high_skip.csv` | 214 |
| NUDT-SIRST | `docs/diagnostics/feature_statistics/nudt_v2_1_shallow_high_skip.csv` | `docs/diagnostics/feature_statistics/summary_nudt_v2_1_shallow_high_skip.csv` | 664 |
| IRSTD-1K | `docs/diagnostics/feature_statistics/irstd_v2_1_shallow_high_skip.csv` | `docs/diagnostics/feature_statistics/summary_irstd_v2_1_shallow_high_skip.csv` | 201 |

本轮配置为 `high_skip_stages: [1, 2]`，因此 summary 中 stage-1/2 的 `stage_used_as_decoder_skip=1`，stage-3/4 为 0；stage-4 high_hat 仅用于诊断和 auxiliary head 监督，不作为 decoder skip。

## 12. MIRFD-Net v2.2 FSRE 实验计划（2026-07-01）

根据 `MIRFD_Net_v2_2_FSRE_frequency_selective_residual_enhancer_plan.md`，v2.2 的目标是把 high branch 从普通卷积 HFE 转向频率选择性残差增强。新增 `FrequencySelectiveResidualEnhancer`，在局部窗口 FFT 中按 radial bands 学习频带权重，并采用 residual-style 输出：

```text
R_freq = LocalBandFFT(R)
high_raw = R + gamma * Proj(R_freq)
```

新增配置开关：

| Switch | Meaning |
|---|---|
| `model.mirfd.high_enhancer_type` | `identity / conv_hfe / freq_window` |
| `model.mirfd.fsre_num_bands` | FSRE radial frequency band 数，默认 4 |
| `model.mirfd.fsre_window_size` | 局部 FFT window size，默认 8 |
| `model.mirfd.fsre_gamma_init` | FSRE residual 增强初始缩放，默认 0.1 |
| `model.stage1_high_enhancer_type` | stage-1 high skip 单独选择 `identity / conv_hfe / freq_window` |
| `model.decoder_high_source` | decoder high skip 使用 `high_raw / high_hat / residual` |

已新增四组 v2.2 配置，每组覆盖 NUAA-SIRST、NUDT-SIRST、IRSTD-1K：

| Experiment | Key idea |
|---|---|
| `v2_2_identity_residual` | decoder 直接使用 residual，验证原始 residual 是否足够 |
| `v2_2_conv_hfe_high_raw` | 普通 Conv-HFE 对照，但 decoder 使用 `high_raw` |
| `v2_2_window_fsre` | stage-1 和 MIRFD stage 都使用 FSRE |
| `v2_2_stage1_identity_stage2_fsre` | stage-1 保留原始 residual，stage-2 使用 FSRE；第一轮主推 |

第一轮优先启动：

```text
configs/mirfd_nuaa_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml
configs/mirfd_nudt_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml
configs/mirfd_irstd_1k_ss2d_v2_2_stage1_identity_stage2_fsre.yaml
```

输出目录：

```text
runs/v2_2_ablation/nuaa_stage1_identity_stage2_fsre
runs/v2_2_ablation/nudt_stage1_identity_stage2_fsre
runs/v2_2_ablation/irstd_stage1_identity_stage2_fsre
```

### 12.1 v2.2 第一轮实验启动记录（2026-07-01）

服务器项目路径：`/DATA20T/bip/cry/code/MIRFD_Net`。

已启动三组单数据集训练，均使用 `v2_2_stage1_identity_stage2_fsre` 配置，即 stage-1 使用 identity residual、MIRFD stage 使用 FSRE、decoder high skip 使用 `high_raw`，并保持 `high_skip_stages: [1, 2]`。

| Dataset | GPU | PID | Config | Output dir | Log |
|---|---:|---:|---|---|---|
| NUAA-SIRST | 0 | 2277341 | `configs/mirfd_nuaa_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml` | `runs/v2_2_ablation/nuaa_stage1_identity_stage2_fsre` | `runs/v2_2_ablation/logs/nuaa_stage1_identity_stage2_fsre.log` |
| NUDT-SIRST | 1 | 2277345 | `configs/mirfd_nudt_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml` | `runs/v2_2_ablation/nudt_stage1_identity_stage2_fsre` | `runs/v2_2_ablation/logs/nudt_stage1_identity_stage2_fsre.log` |
| IRSTD-1K | 2 | 2281979 | `configs/mirfd_irstd_1k_ss2d_v2_2_stage1_identity_stage2_fsre.yaml` | `runs/v2_2_ablation/irstd_stage1_identity_stage2_fsre` | `runs/v2_2_ablation/logs/irstd_stage1_identity_stage2_fsre.log` |

启动检查：

| Dataset | Latest checked epoch | IoU | nIoU | Pd | Fa | Note |
|---|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | 48 | 0.4463 | 0.5246 | 0.9354 | 0.000381 | early status only, not final |
| NUDT-SIRST | 27 | 0.3644 | 0.4069 | 0.9291 | 0.000455 | early status only, not final |
| IRSTD-1K | 6 | 0.0194 | 0.0412 | 0.1735 | 0.005606 | restarted after cleaning a CRLF-tainted output directory |

上述数值仅用于确认训练和验证循环正常运行，不能作为最终性能对比。最终比较仍以各 run 的 `best.pt` / `best_iou.pt` 和完整测试集评估为准。

### 12.2 v2.2 第一轮完成结果（2026-07-01）

三组 `v2_2_stage1_identity_stage2_fsre` 已完成 500 epoch。该组配置为 stage-1 identity residual、MIRFD stage 使用 FSRE、decoder high skip 使用 `high_raw`，训练策略沿用 SCTransNet-style preprocessing、AdamW、cosine warmup、关闭 spectral loss。

| Dataset | Run | Best epoch | IoU | nIoU | Dice | Precision | Recall | Pd | Fa | Latest epoch IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `nuaa_stage1_identity_stage2_fsre` | 136 | 0.6900 | 0.6881 | 0.8165 | 0.7909 | 0.8439 | 0.9582 | 0.000054 | 0.6137 |
| NUDT-SIRST | `nudt_stage1_identity_stage2_fsre` | 500 | 0.8756 | 0.8959 | 0.9337 | 0.9348 | 0.9325 | 0.9841 | 0.000010 | 0.8756 |
| IRSTD-1K | `irstd_stage1_identity_stage2_fsre` | 338 | 0.5968 | 0.5440 | 0.7475 | 0.6714 | 0.8430 | 0.8503 | 0.000027 | 0.5452 |

与上一轮 v2.1 shallow high skip / 当前全局最佳对比：

| Dataset | v2.1 shallow IoU | Previous global best IoU | v2.2 FSRE IoU | vs v2.1 shallow | vs previous global best | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | 0.7267 | 0.7452 | 0.6900 | -0.0367 | -0.0552 | 明显下降，不建议继续该结构作为 NUAA 主线 |
| NUDT-SIRST | 0.8618 | 0.8696 | 0.8756 | +0.0138 | +0.0060 | 有效，刷新 NUDT 当前最佳 |
| IRSTD-1K | 0.6290 | 0.6290 | 0.5968 | -0.0322 | -0.0322 | 下降，不如 v2.1 shallow |

结论：

1. FSRE 不是跨数据集稳定正收益；当前只在 NUDT-SIRST 上带来明确提升。
2. NUDT-SIRST 的 best epoch 出现在 500，说明该结构在 NUDT 上仍有继续训练或微调学习率尾段的空间。
3. NUAA-SIRST best 出现在 epoch 136，后续回落明显，说明 `stage1 identity + FSRE + high_raw` 可能引入了不稳定高频响应或过拟合。
4. IRSTD-1K 虽然中后期达到 0.5968，但仍低于 v2.1 shallow 的 0.6290，暂不替换 IRSTD 主配置。
5. 当前最佳结果表需要更新：NUDT-SIRST 的全局最佳从 `nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3` 替换为 `v2_2_ablation/nudt_stage1_identity_stage2_fsre`；NUAA-SIRST 和 IRSTD-1K 保持原最佳。

### 12.3 v2.2 FSRE 频域与目标选择性诊断（2026-07-01）

已生成 v2.2 FSRE 的测试集内部特征统计和可视化：

| Dataset | Raw CSV | Summary CSV | Visualization |
|---|---|---|---|
| NUAA-SIRST | `docs/diagnostics/feature_statistics/nuaa_v2_2_stage1_identity_stage2_fsre.csv` | `docs/diagnostics/feature_statistics/summary_nuaa_v2_2_stage1_identity_stage2_fsre.csv` | `docs/visualizations/v2_2_fsre_feature_diagnostics/nuaa_v2_diagnostic/contact_sheet.png` |
| NUDT-SIRST | `docs/diagnostics/feature_statistics/nudt_v2_2_stage1_identity_stage2_fsre.csv` | `docs/diagnostics/feature_statistics/summary_nudt_v2_2_stage1_identity_stage2_fsre.csv` | `docs/visualizations/v2_2_fsre_feature_diagnostics/nudt_v2_diagnostic/contact_sheet.png` |
| IRSTD-1K | `docs/diagnostics/feature_statistics/irstd_v2_2_stage1_identity_stage2_fsre.csv` | `docs/diagnostics/feature_statistics/summary_irstd_v2_2_stage1_identity_stage2_fsre.csv` | `docs/visualizations/v2_2_fsre_feature_diagnostics/irstd_v2_diagnostic/contact_sheet.png` |

与 v2.1 shallow high skip 对比，关键现象如下：

| Dataset | Stage | `R_high_high_raw` change | `high_raw_fg_bg` change | `high_hat_fg_bg` change | `gate_fg_minus_bg` change | `false_alarm_rate` change | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | 2 | +0.1087 | -1.0111 | -0.8256 | -0.0019 | +0.0467 | 高频能量增加，但目标选择性下降，误警上升 |
| NUDT-SIRST | 3 | +0.0667 | +0.8824 | +0.7707 | +0.0625 | +0.0060 | 深层 high response 更目标相关，解释 NUDT 提升 |
| NUDT-SIRST | 4 | +0.0718 | +0.1530 | +0.2498 | +0.0756 | +0.0060 | gate 对目标区域更正向，深层语义更干净 |
| IRSTD-1K | 2 | +0.0679 | +0.7663 | +1.2234 | +0.1400 | -0.0050 | stage-2 FSRE 明显增强目标相关高频 |
| IRSTD-1K | 4 | +0.1184 | -0.1180 | -0.1468 | +0.0190 | -0.0050 | 深层高频能量增加但目标选择性下降，限制最终 IoU |

可视化结论：

1. FSRE 确实让 `high_raw/high_hat` 的频谱高频成分更强，尤其 stage-2/3/4 的 FFT 图中高频扩展更明显。
2. “高频更强”不等于“小目标更好”。NUAA 的 stage-2 高频增强同时覆盖背景纹理，`high_raw_fg_bg` 反而从 4.1339 降到 3.1228，false alarm rate 从 0.0654 升到 0.1121。
3. NUDT 的收益更合理：stage-3/4 的 `high_raw_fg_bg` 和 `high_hat_fg_bg` 同时提升，gate 的 foreground-background 差值由负或弱正变为更明显正值，因此 FSRE 后的高频更像目标相关细节，而不是背景噪声。
4. IRSTD 的 stage-2 目标选择性提升明显，但 stage-4 的 `high_hat_fg_bg` 降到 0.8338，说明深层高频仍可能偏向背景结构；这解释了诊断指标略有改善但最终 best IoU 仍不如 v2.1 shallow。
5. 下一步不建议简单放大 FSRE，而应做 stage-aware FSRE：NUDT 可保留 stage-3/4 FSRE，NUAA 应降低或关闭 stage-2 FSRE，IRSTD 应限制 deep FSRE 或只保留 stage-2 FSRE 并抑制 stage-4 high response。

## 13. MIRFD-Net v2.3 high raw / residual fusion 消融计划（2026-07-01）

根据 `MIRFD_Net_v2_3_high_raw_fusion_ablation_plan.md`，v2.3 重点验证一个机制问题：小目标信息可能已经存在于 `residual/high_raw`，但 `gate -> high_hat` 调制不稳定，导致 block 主路径把背景高频也融合进 encoder 表示。

v2.3 新增开关：

| Switch | Values | Meaning |
|---|---|---|
| `model.mirfd.block_fusion_high_source` | `high_hat / high_raw / residual` | MIRFD Block 主路径 `Fuse(low, high)` 使用哪个 high 分支 |
| `model.mirfd.gate_mode` | 新增 `none` | `none` 时不构建 gate，`high_hat == high_raw`，但仍输出全 1 gate map 用于诊断兼容 |

`features["high"]` 和 `features["high_for_fusion"]` 现在表示实际进入 block fusion 的 high feature；`features["high_raw"]` 和 `features["high_hat"]` 仍显式保留。诊断脚本同步新增：

```text
R_high_high_for_fusion
high_for_fusion_fg_bg
block_fusion_high_source
```

实验矩阵：

| Experiment | Config suffix | block fusion source | gate mode | decoder high source | Purpose |
|---|---|---|---|---|---|
| B | `block_high_raw_gate_none` | `high_raw` | `none` | `high_raw` | 验证 gate/high_hat 是否是主要瓶颈 |
| C | `block_residual_gate_none` | `residual` | `none` | `high_raw` | 验证 encoder 主路径是否应更保守地使用 residual |
| D | `block_high_raw_gate_enhance` | `high_raw` | `enhance` | `high_raw` | 保留 gate 诊断，但不让 high_hat 进入 block 主路径 |

已新增 9 个配置文件，覆盖 NUAA-SIRST、NUDT-SIRST、IRSTD-1K：

```text
configs/mirfd_nuaa_sirst_ss2d_v2_3_block_high_raw_gate_none.yaml
configs/mirfd_nuaa_sirst_ss2d_v2_3_block_residual_gate_none.yaml
configs/mirfd_nuaa_sirst_ss2d_v2_3_block_high_raw_gate_enhance.yaml
configs/mirfd_nudt_sirst_ss2d_v2_3_block_high_raw_gate_none.yaml
configs/mirfd_nudt_sirst_ss2d_v2_3_block_residual_gate_none.yaml
configs/mirfd_nudt_sirst_ss2d_v2_3_block_high_raw_gate_enhance.yaml
configs/mirfd_irstd_1k_ss2d_v2_3_block_high_raw_gate_none.yaml
configs/mirfd_irstd_1k_ss2d_v2_3_block_residual_gate_none.yaml
configs/mirfd_irstd_1k_ss2d_v2_3_block_high_raw_gate_enhance.yaml
```

输出目录统一为：

```text
runs/v2_3_ablation/<dataset>_block_high_raw_gate_none
runs/v2_3_ablation/<dataset>_block_residual_gate_none
runs/v2_3_ablation/<dataset>_block_high_raw_gate_enhance
```

优先观察：

1. 如果 B 优于 v2.2 A，说明 `gate/high_hat` 是主要瓶颈，direct `high_raw` fusion 更合理。
2. 如果 C 优于 B，说明 `high_raw` 更适合 decoder skip，但 encoder 主路径应使用更保守的 `residual`。
3. 如果 D 优于 A 但弱于 B，说明 gate 计算可以保留用于诊断或辅助监督，但不应参与主路径 fusion。

### 13.1 v2.3 消融启动记录（2026-07-01）

已按 `MIRFD_Net_v2_3_high_raw_fusion_ablation_plan.md` 完成代码接入并启动实验。实现层面新增：

1. `model.mirfd.block_fusion_high_source`，支持 `high_hat / high_raw / residual`，控制 MIRFD Block 内部 `Fuse(low, high)` 使用的高频源。
2. `model.mirfd.gate_mode: none`，关闭 gate 模块时令 `high_hat == high_raw`，并输出全 1 gate map 以兼容诊断脚本。
3. `features["high_for_fusion"]` 和 `features["block_fusion_high_source"]`，诊断脚本同步新增 `R_high_high_for_fusion` 与 `high_for_fusion_fg_bg`。
4. FSRE 的局部 FFT、频带能量和 band MLP 计算在 AMP 下固定为 fp32，避免 `ComplexHalf` FFT warning 和半精度频域数值不稳定。

服务器路径：`/DATA20T/bip/cry/code/MIRFD_Net`。启动前已通过服务器端 `tests/smoke_test.py`，并用 CUDA+AMP 前向验证：

```text
max_abs(high_for_fusion - high_raw) = 0.0
complex_half_warnings = 0
```

本轮实验 launcher PID：`2534612`，启动时间：`2026-07-01 17:21:15`。使用 GPU `0 1 2 3 4 5 6`，前 7 组并行，剩余 2 组由同一 launcher 排队执行。

| Dataset | Variant | GPU status | Config | Output dir | Log |
|---|---|---:|---|---|---|
| NUAA-SIRST | `block_high_raw_gate_none` | 0 running | `configs/mirfd_nuaa_sirst_ss2d_v2_3_block_high_raw_gate_none.yaml` | `runs/v2_3_ablation/nuaa_block_high_raw_gate_none` | `runs/v2_3_ablation/logs/nuaa_block_high_raw_gate_none.log` |
| NUDT-SIRST | `block_high_raw_gate_none` | 1 running | `configs/mirfd_nudt_sirst_ss2d_v2_3_block_high_raw_gate_none.yaml` | `runs/v2_3_ablation/nudt_block_high_raw_gate_none` | `runs/v2_3_ablation/logs/nudt_block_high_raw_gate_none.log` |
| IRSTD-1K | `block_high_raw_gate_none` | 2 running | `configs/mirfd_irstd_1k_ss2d_v2_3_block_high_raw_gate_none.yaml` | `runs/v2_3_ablation/irstd_block_high_raw_gate_none` | `runs/v2_3_ablation/logs/irstd_block_high_raw_gate_none.log` |
| NUAA-SIRST | `block_residual_gate_none` | 3 running | `configs/mirfd_nuaa_sirst_ss2d_v2_3_block_residual_gate_none.yaml` | `runs/v2_3_ablation/nuaa_block_residual_gate_none` | `runs/v2_3_ablation/logs/nuaa_block_residual_gate_none.log` |
| NUDT-SIRST | `block_residual_gate_none` | 4 running | `configs/mirfd_nudt_sirst_ss2d_v2_3_block_residual_gate_none.yaml` | `runs/v2_3_ablation/nudt_block_residual_gate_none` | `runs/v2_3_ablation/logs/nudt_block_residual_gate_none.log` |
| IRSTD-1K | `block_residual_gate_none` | 5 running | `configs/mirfd_irstd_1k_ss2d_v2_3_block_residual_gate_none.yaml` | `runs/v2_3_ablation/irstd_block_residual_gate_none` | `runs/v2_3_ablation/logs/irstd_block_residual_gate_none.log` |
| NUAA-SIRST | `block_high_raw_gate_enhance` | 6 running | `configs/mirfd_nuaa_sirst_ss2d_v2_3_block_high_raw_gate_enhance.yaml` | `runs/v2_3_ablation/nuaa_block_high_raw_gate_enhance` | `runs/v2_3_ablation/logs/nuaa_block_high_raw_gate_enhance.log` |
| NUDT-SIRST | `block_high_raw_gate_enhance` | queued | `configs/mirfd_nudt_sirst_ss2d_v2_3_block_high_raw_gate_enhance.yaml` | `runs/v2_3_ablation/nudt_block_high_raw_gate_enhance` | `runs/v2_3_ablation/logs/nudt_block_high_raw_gate_enhance.log` |
| IRSTD-1K | `block_high_raw_gate_enhance` | queued | `configs/mirfd_irstd_1k_ss2d_v2_3_block_high_raw_gate_enhance.yaml` | `runs/v2_3_ablation/irstd_block_high_raw_gate_enhance` | `runs/v2_3_ablation/logs/irstd_block_high_raw_gate_enhance.log` |

旧的 17:18 试跑在发现 FSRE under AMP 会触发 `ComplexHalf` warning 后已停止，并清理了 `runs/v2_3_ablation` 目录后重新启动；最终对比应只使用 17:21 后的新日志和 checkpoint。

### 13.2 v2.3 完成结果与高低频诊断（2026-07-02）

9 组 v2.3 消融已全部完成 500 epoch，并已生成内部特征统计和高低频可视化。统计输出位于：

```text
docs/diagnostics/feature_statistics/v2_3/
docs/visualizations/v2_3_residual_gate_none_feature_diagnostics/
```

其中：

```text
docs/diagnostics/feature_statistics/v2_3/v2_3_residual_gate_none_key_metrics.csv
docs/diagnostics/feature_statistics/v2_3/v2_3_variant_key_metrics.csv
docs/diagnostics/feature_statistics/v2_3/v2_3_residual_gate_none_key_metrics.png
docs/visualizations/v2_3_residual_gate_none_feature_diagnostics/nuaa_residual_gate_none/contact_sheet.png
docs/visualizations/v2_3_residual_gate_none_feature_diagnostics/nudt_residual_gate_none/contact_sheet.png
docs/visualizations/v2_3_residual_gate_none_feature_diagnostics/irstd_residual_gate_none/contact_sheet.png
```

训练结果如下：

| Dataset | Variant | Best IoU | Epoch | nIoU | Dice | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `block_high_raw_gate_none` | 0.7300 | 255 | 0.7177 | 0.8440 | 0.9658 | 0.000024 |
| NUAA-SIRST | `block_residual_gate_none` | **0.7307** | 337 | 0.7159 | 0.8444 | 0.9506 | 0.000029 |
| NUAA-SIRST | `block_high_raw_gate_enhance` | 0.7172 | 224 | 0.6992 | 0.8353 | 0.9202 | 0.000017 |
| NUDT-SIRST | `block_high_raw_gate_none` | 0.8597 | 436 | 0.8800 | 0.9245 | 0.9820 | 0.000013 |
| NUDT-SIRST | `block_residual_gate_none` | **0.8746** | 457 | 0.8962 | 0.9331 | 0.9831 | 0.000011 |
| NUDT-SIRST | `block_high_raw_gate_enhance` | 0.8669 | 435 | 0.8839 | 0.9287 | 0.9799 | 0.000008 |
| IRSTD-1K | `block_high_raw_gate_none` | 0.5991 | 248 | 0.5191 | 0.7493 | 0.8367 | 0.000017 |
| IRSTD-1K | `block_residual_gate_none` | **0.6114** | 264 | 0.5571 | 0.7589 | 0.8469 | 0.000024 |
| IRSTD-1K | `block_high_raw_gate_enhance` | 0.6004 | 356 | 0.5361 | 0.7503 | 0.8537 | 0.000024 |

定量统计说明：下表中的 `sample mean IoU` 是测试样本级 IoU 的均值，用于和 false alarm / feature statistics 对齐，不等同于训练日志中的全数据集 IoU。

| Dataset | Best v2.3 variant | Stage | Decoder skip | `R_high(low)` | `R_high(residual)` | `R_high(fusion)` | `residual fg/bg` | `high_raw fg/bg` | `fusion fg/bg` | FA rate | sample mean IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `residual_gate_none` | 1 | 1 | 0.510 | 0.638 | 0.638 | 13.83 | 13.83 | 13.83 | 0.051 | 0.716 |
| NUAA-SIRST | `residual_gate_none` | 2 | 1 | 0.827 | 0.793 | 0.793 | 4.74 | 4.22 | 4.74 | 0.051 | 0.716 |
| NUAA-SIRST | `residual_gate_none` | 3 | 0 | 0.760 | 0.737 | 0.737 | 2.16 | 3.06 | 2.16 | 0.051 | 0.716 |
| NUAA-SIRST | `residual_gate_none` | 4 | 0 | 0.680 | 0.610 | 0.610 | 0.86 | 1.17 | 0.86 | 0.051 | 0.716 |
| NUDT-SIRST | `residual_gate_none` | 1 | 1 | 0.609 | 0.677 | 0.677 | 5.82 | 5.82 | 5.82 | 0.026 | 0.896 |
| NUDT-SIRST | `residual_gate_none` | 2 | 1 | 0.821 | 0.794 | 0.794 | 5.32 | 4.43 | 5.32 | 0.026 | 0.896 |
| NUDT-SIRST | `residual_gate_none` | 3 | 0 | 0.688 | 0.685 | 0.685 | 1.84 | 2.47 | 1.84 | 0.026 | 0.896 |
| NUDT-SIRST | `residual_gate_none` | 4 | 0 | 0.546 | 0.465 | 0.465 | 0.98 | 1.07 | 0.98 | 0.026 | 0.896 |
| IRSTD-1K | `residual_gate_none` | 1 | 1 | 0.582 | 0.690 | 0.690 | 12.86 | 12.86 | 12.86 | 0.080 | 0.557 |
| IRSTD-1K | `residual_gate_none` | 2 | 1 | 0.808 | 0.775 | 0.775 | 4.29 | 4.52 | 4.29 | 0.080 | 0.557 |
| IRSTD-1K | `residual_gate_none` | 3 | 0 | 0.788 | 0.758 | 0.758 | 3.19 | 4.06 | 3.19 | 0.080 | 0.557 |
| IRSTD-1K | `residual_gate_none` | 4 | 0 | 0.702 | 0.672 | 0.672 | 0.83 | 0.88 | 0.83 | 0.080 | 0.557 |

诊断结论：

1. 三个数据集的最佳 v2.3 变体都是 `block_residual_gate_none`。这说明 v2.3 的主要收益不是来自 gate，而是来自让 block 主路径使用更保守的 `residual` 作为 high fusion source。
2. Stage-1 和 stage-2 是最干净的局部小目标高频来源。三组数据中 stage-1 residual 的 `fg/bg` 均明显大于 1；stage-2 residual/fusion 的 `fg/bg` 也保持在 4.29-5.32 左右。
3. Stage-4 的 high/fusion target selectivity 明显退化。`fusion fg/bg` 在 NUAA、NUDT、IRSTD 上分别为 0.86、0.98、0.83，接近或低于 1，说明深层 high response 更容易混入背景结构。这继续支持当前只使用 stage-1/2 decoder high skip，而不启用 stage-3/4 high skip。
4. `high_raw` 在 stage-3 有时比 residual 更 target-selective，例如 NUAA stage-3 从 2.16 升到 3.06，NUDT stage-3 从 1.84 升到 2.47，IRSTD stage-3 从 3.19 升到 4.06。但直接让 high_raw 进入 block fusion 并没有带来更高 IoU，说明局部 feature 的 fg/bg 提升不等价于更好的全局 encoder 表示；增强后的 high_raw 仍可能携带背景纹理。
5. `gate_mode: enhance` 没有稳定收益。其 gate foreground-background 差异很小，部分 stage 还为负；同时 `high_hat fg/bg` 通常没有稳定高于 `high_raw fg/bg`。因此当前 gate 还不能证明具备足够 target-aware 的调制能力。
6. 可视化与统计一致：contact sheet 中 stage-1/2 的 residual/fusion 热力图更集中于目标附近，而 stage-3/4 的结构图和 FFT 图中背景纹理、边缘和条带响应更强。NUAA 与 IRSTD 的深层 high response 尤其容易覆盖背景结构，这解释了它们距离最优结果仍有差距。

下一步建议：

1. 主线优先保留 `block_fusion_high_source: residual` 与 `gate_mode: none`，继续围绕 stage-1/2 high skip 做轻量增强。
2. 不建议简单恢复 deep high skip；如果后续使用 stage-3 high 信息，应先加 target-aware 筛选或 foreground-guided regularization。
3. FSRE 可以保留在 high_raw 诊断分支中，但不宜直接作为 block 主路径 fusion 源；更合理的是只在 stage-2 或 dataset-specific 设置中启用。

## 14. MIRFD-Net v2.4 FFC-style Fourier high enhancer（2026-07-02）

根据 `E:\code\SCTransNet-main\SCTransNet-main` 中的 FFC 实现，对 MIRFD high branch 做一次更接近 FFC 的频域增强改造。本轮没有直接把整个 FFC U-Net 或 SCTransNet 主干搬进 MIRFD，因为 MIRFD 的核心变量仍然是 Mamba/SS2D 产生的 `low` 与 `residual = F - low`；直接替换主干会破坏已有 v2.3 诊断结论的可比性。

### 参考了什么

主要参考文件：

```text
E:\code\SCTransNet-main\SCTransNet-main\model\lama_ffc.py
E:\code\SCTransNet-main\SCTransNet-main\model\u2net.py
```

具体参考点：

1. `FourierUnit`：使用 `torch.fft.rfftn` 将空间特征变到频域，把 real/imag 作为通道维度进行 1x1 卷积、BN、ReLU，再用 `irfftn` 回到空间域。
2. `SpectralTransform`：先用 1x1 conv 压缩/整理通道，再进入 FourierUnit，最后再投影回输出通道；这说明 FFC 的频域分支不是简单 FFT 可视化，而是可学习的频域通道混合。
3. `FFC`：将特征拆成 local/global 两部分，local 用普通卷积，global 用 `SpectralTransform`，再通过 `l2l/l2g/g2l/g2g` 路径融合。MIRFD 这里没有照搬通道拆分，而是把 residual high branch 视为需要频域增强的 global-high 分支。
4. `SpectralHighFreqGate` 和 `FourierUnit_SpectralHighFreqGate_FeatureView`：在频域卷积后，根据高频区域幅值统计预测 `1 + tanh(.)` 的通道 gate，且最后一层零初始化，使初始 gate 接近 1，训练更稳。

### 调整了什么

新增代码：

```text
mirfd/models/frequency_enhancer.py
  - FFCStyleHighFreqGate
  - FFCFrequencyResidualEnhancer
```

`FFCFrequencyResidualEnhancer` 的路径为：

```text
R
├─ local branch: depthwise ConvNormAct(R)
└─ Fourier branch:
   rfftn(R)
   real/imag stack as channels
   1x1 conv + BN + ReLU
   optional high-frequency gate
   irfftn(...)

high_raw = R + gamma * Fuse(local, Fourier)
```

与原 v2.2 FSRE 的区别：

| Module | Frequency granularity | Learnable operation | Output role |
|---|---|---|---|
| FSRE | local window FFT + radial bands | band-wise weights | `R + gamma * local-frequency response` |
| FFC-style enhancer | global rFFT feature map | real/imag 1x1 conv + high-frequency gate | `R + gamma * Fuse(local, global Fourier response)` |

新增配置开关：

```yaml
model:
  mirfd:
    high_enhancer_type: ffc
    ffc_fft_norm: ortho
    ffc_gamma_init: 0.1
    ffc_use_highfreq_gate: true
    ffc_highfreq_threshold: 0.5
    ffc_gate_reduction: 4
    ffc_local_kernel: 3
```

已接入的构建路径：

```text
mirfd/models/mirfd_block.py::build_high_enhancer(...)
mirfd/models/mirfd_net.py::build_model(...)
mirfd/models/__init__.py
tests/smoke_test.py
```

### v2.4 实验设计

为了和 v2.3 最优结论保持可比，本轮只替换 high enhancer：

```text
block_fusion_high_source: residual
gate_mode: none
decoder_high_source: high_raw
stage1_high_enhancer_type: identity
high_skip_stages: [1, 2]
high_enhancer_type: ffc
```

也就是说：

1. MIRFD Block 主路径仍用 `residual` 做 high fusion，避免 high_raw 直接污染 encoder。
2. decoder stage-2 high skip 使用 FFC 学到的 `high_raw`，用于验证 FFC 是否能比 FSRE 学到更适合小目标恢复的高频特征。
3. stage-1 仍保持 identity residual，避免浅层最干净的 high skip 被额外频域模块扰动。

新增配置：

```text
configs/mirfd_nuaa_sirst_ss2d_v2_4_ffc_residual_gate_none.yaml
configs/mirfd_nudt_sirst_ss2d_v2_4_ffc_residual_gate_none.yaml
configs/mirfd_irstd_1k_ss2d_v2_4_ffc_residual_gate_none.yaml
```

新增启动脚本：

```text
scripts/run_v2_4_ffc_experiments.sh
```

预期判断：

1. 如果 v2.4 FFC 高于 v2.3 residual_gate_none，说明全局 Fourier real/imag mixing 比 FSRE 的 local band weighting 更适合当前 high_raw decoder skip。
2. 如果 v2.4 FFC 只提升 NUDT，不提升 NUAA/IRSTD，说明全局频域增强仍存在 dataset-specific 背景纹理放大问题。
3. 如果 v2.4 FFC 下降，说明当前 MIRFD 的收益主要来自 residual 的保守注入，而不是更强的 high_raw 频域增强；后续应考虑 foreground-guided 或 mask-aware spectral gate，而不是继续放大频域分支。
## 15. MIRFD-Net v2.4 FFC 内部特征统计与高低频可视化诊断（2026-07-02）

诊断对象为 v2.4 FFC-style Fourier high enhancer 三组 `best_iou.pt`：

```text
runs/v2_4_ffc_ablation/nuaa_ffc_residual_gate_none/best_iou.pt
runs/v2_4_ffc_ablation/nudt_ffc_residual_gate_none/best_iou.pt
runs/v2_4_ffc_ablation/irstd_ffc_residual_gate_none/best_iou.pt
```

输出文件：

```text
docs/diagnostics/feature_statistics/v2_4_ffc/
docs/visualizations/v2_4_ffc_feature_diagnostics/
```

每个数据集均生成了全量 test CSV 统计，以及 20 张低频/高频/FFT 诊断图和 `contact_sheet.png`。统计中的 `pred_iou` 与 `pred_has_false_alarm` 是最终预测的样本级指标，在不同 stage 行中重复记录，并不是 stage 自身输出。

### v2.4 FFC best_iou 结果

| Dataset | best epoch | IoU | nIoU | Dice | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 187 | 0.7391 | 0.7187 | 0.8500 | 0.9582 | 0.000012 |
| NUDT-SIRST | 474 | 0.8468 | 0.8713 | 0.9170 | 0.9767 | 0.000024 |
| IRSTD-1K | 325 | 0.5956 | 0.5571 | 0.7466 | 0.8605 | 0.000033 |

### 内部特征统计摘要

| Dataset | Stage | Decoder skip | R_low | R_residual | R_high_raw | residual-low | high_raw-residual fg/bg |
|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 1 | 1 | 0.5292 | 0.6420 | 0.6420 | +0.1129 | +0.0000 |
| NUAA-SIRST | 2 | 1 | 0.8304 | 0.8022 | 0.7653 | -0.0283 | +1.3689 |
| NUAA-SIRST | 3 | 0 | 0.5818 | 0.6148 | 0.6012 | +0.0331 | +0.1867 |
| NUAA-SIRST | 4 | 0 | 0.6466 | 0.5562 | 0.5193 | -0.0903 | +0.0735 |
| NUDT-SIRST | 1 | 1 | 0.5746 | 0.6659 | 0.6659 | +0.0913 | +0.0000 |
| NUDT-SIRST | 2 | 1 | 0.8071 | 0.7761 | 0.7223 | -0.0310 | -0.5519 |
| NUDT-SIRST | 3 | 0 | 0.6809 | 0.5660 | 0.6062 | -0.1148 | +0.1184 |
| NUDT-SIRST | 4 | 0 | 0.6116 | 0.5312 | 0.5167 | -0.0803 | -0.0070 |
| IRSTD-1K | 1 | 1 | 0.5807 | 0.6768 | 0.6768 | +0.0961 | +0.0000 |
| IRSTD-1K | 2 | 1 | 0.8259 | 0.8041 | 0.7651 | -0.0218 | -1.3463 |
| IRSTD-1K | 3 | 0 | 0.7277 | 0.6934 | 0.6942 | -0.0344 | +0.0464 |
| IRSTD-1K | 4 | 0 | 0.6965 | 0.5549 | 0.5355 | -0.1416 | -0.0141 |

### 诊断结论

1. Stage-1 residual 是最稳定的高频目标分支。三个数据集 stage-1 均满足 `R_high_residual > R_high_low`，并且 `residual_fg_bg` 明显高于 `low_fg_bg`。这说明浅层 `F - low` 对小目标有清晰选择性，继续保留 stage-1 identity residual skip 是合理的。
2. Stage-2 FFC 的效果具有数据集差异。NUAA 上 `high_raw_fg_bg - residual_fg_bg = +1.3689`，说明 FFC high_raw 明显增强了目标选择性；但 NUDT 为 `-0.5519`，IRSTD 为 `-1.3463`，说明 FFC 在这两个数据集上反而削弱了 stage-2 residual 的目标/背景区分。
3. Stage-2 的频谱高频比例并没有被 FFC 放大，反而被压低。三个数据集 stage-2 都有 `R_high_high_raw < R_high_residual`，说明 FFC-style Fourier branch 更像是在做可学习频域重整和平滑，而不是单纯增加高频能量。
4. Stage-3/4 不作为 decoder high skip 是正确的。统计中 stage-3/4 的 `fg/bg` 普遍接近 1，尤其 stage-4 在 NUDT 和 IRSTD 上低于或接近 1，说明深层 residual/high_raw 更容易混入背景语义，不适合直接作为 decoder high skip。
5. 当前 v2.4 设置为 `gate_mode: none`，因此 `gate_fg_minus_bg` 约为 0，`high_hat == high_raw`。这组实验主要验证 FFC high_raw 本身，不验证 target-aware gate。
6. 可视化上，FFT 图显示 stage-1/2 的 residual 和 high_raw 对中心小目标区域更集中；stage-3/4 的 FFT 响应更容易呈现大面积条纹和背景纹理扩散。这与 CSV 中 deep stage `fg/bg` 下降一致。

后续改进方向应避免继续无约束放大深层高频。更合理的方向是：

```text
stage-1 identity residual skip 保留
stage-2 FFC/FSRE 只在 foreground-aware 或 mask-aware gate 下使用
stage-3/4 high_raw 不进入 decoder skip
对 IRSTD 单独加入背景纹理抑制或 target-aware spectral gate
```
