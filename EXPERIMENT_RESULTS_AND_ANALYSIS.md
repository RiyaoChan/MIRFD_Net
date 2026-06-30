# MIRFD-Net 实验结果与模型分析记录

记录时间：2026-06-29  
最近更新：2026-06-30
服务器路径：`/DATA20T/bip/cry/code/MIRFD_Net`  
数据集根目录：`/DATA20T/bip/cry/code/SIRST-5K-main/dataset/`

本文记录当前 MIRFD-Net 在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上的训练结果、训练策略变化、模型分支诊断，以及后续改进方向。

## 1. 当前最佳结果

三组最有效的配置都采用 SCTransNet-style 数据预处理，包括 raw intensity mean/std 标准化、`256x256` 正样本优先 crop、翻转/转置增强、AdamW、cosine warmup，以及 centroid Pd/Fa 统计。

| Dataset | Best run | Config | Best epoch | IoU | nIoU | Dice | Precision | Recall | Pd | Fa | Compared with previous SS2D |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `nuaa_sirst_ss2d_sctrans_adamw_lr1e3` | `configs/mirfd_nuaa_sirst_ss2d_sctrans_adamw_lr1e3.yaml` | 374 | 0.7452 | 0.7184 | 0.8540 | 0.8443 | 0.8639 | 0.9696 | 0.000017 | +0.0320 IoU |
| NUDT-SIRST | `nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3` | `configs/mirfd_nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3.yaml` | 441 | 0.8696 | 0.8926 | 0.9302 | 0.9329 | 0.9276 | 0.9799 | 0.000011 | +0.0718 IoU |
| IRSTD-1K | `v2_ablation/irstd_v2_no_spectral` | `configs/mirfd_irstd_1k_ss2d_v2_no_spectral.yaml` | 160 | 0.6129 | 0.5311 | 0.7600 | 0.6965 | 0.8364 | 0.8571 | 0.000018 | +0.0190 IoU |

对应 checkpoint：

| Dataset | Best checkpoint |
|---|---|
| NUAA-SIRST | `runs/nuaa_sirst_ss2d_sctrans_adamw_lr1e3/best.pt` |
| NUDT-SIRST | `runs/nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3/best.pt` |
| IRSTD-1K | `runs/v2_ablation/irstd_v2_no_spectral/best.pt` |

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
