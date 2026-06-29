# MIRFD-Net v2 改进计划：给 Codex 的代码修改说明

> 任务：红外小目标分割（NUAA-SIRST、NUDT-SIRST、IRSTD-1K）  
> 当前核心：Mamba-induced Residual Frequency Decoupling  
> 目标：根据当前实验诊断，升级 MIRFD Block、gate、high skip、spectral loss 和训练稳定性。

实现状态（2026-06-29）：

- 已实现：LowSmooth、high residual `concat_proj/add`、增强型 gate、Stage1 high skip、`spectral_high_target`、NaN 保护、多指标 checkpoint、v2 FFT 诊断、`pyramid_avgpool` residual ablation；真正 Haar wavelet high-frequency 尚未实现。
- 已新增配置：`configs/mirfd_nuaa_sirst_ss2d_v2.yaml`、`configs/mirfd_nudt_sirst_ss2d_v2.yaml`、`configs/mirfd_irstd_1k_ss2d_v2.yaml`。
- 默认兼容：旧配置的模型结构默认仍使用 v1 行为，即 `use_low_smooth=false`、`high_residual_mode=hfe`、`gate_mode=suppress`、`use_stage1_high_skip=false`；loss 的默认高频频谱目标已改为 `high_raw`，严格复现实验前约束需显式设置 `spectral_high_target: high` 或 `high_hat`。

---

## 1. 当前实验诊断

当前结果说明 MIRFD-Net 的方向是有潜力的，但结构还没有充分发挥核心 idea。

主要现象：

1. **NUAA-SIRST 和 NUDT-SIRST 提升明显**，说明 Mamba-induced residual 方向有效。
2. **IRSTD-1K 提升有限，nIoU / Pd 不够稳定**，说明复杂场景下目标级响应还不足。
3. **high 分支对目标有响应**，但 gate 在部分 stage 可能出现目标区域响应低于背景的问题。
4. **low/high 频谱差异存在但不够强**，说明 `low = align(SS2D(F))` 还不够稳定地表现为低频语义近似。
5. **Stage 1 没有 high residual skip**，而红外小目标高度依赖浅层高分辨率细节。

下一轮不要优先继续调学习率和 batch size，应优先做 MIRFD Block v2。

---

## 2. MIRFD Block v2 总体逻辑

当前 v1 逻辑大致是：

```python
fm = self.mamba(self.norm(x))
low = self.align(fm)
residual = x - low
high = self.hfe(residual)
gate = self.gate(low, residual)
high_hat = gate * high
out = self.fuse(torch.cat([low, high_hat], dim=1)) + x
```

v2 推荐改为：

```python
fm = self.mamba(self.norm(x))
low0 = self.align(fm)
low = self.low_smooth(low0)  # optional

residual = x - low

hfe_out = self.hfe(residual)
high_raw = self.high_proj(torch.cat([residual, hfe_out], dim=1))
# optional simpler mode:
# high_raw = residual + self.hfe(residual)

gate = self.gate(low, residual)

alpha = torch.clamp(self.gate_alpha, 0.0, 2.0)
high_hat = (1.0 + alpha * gate) * high_raw

out = self.fuse(torch.cat([low, high_hat], dim=1)) + x
```

返回特征建议改成：

```python
features = {
    "low0": low0,
    "low": low,
    "residual": residual,
    "high_raw": high_raw,
    "high_hat": high_hat,
    "gate": gate,
}
```

含义：

- `low0`：SS2D 输出对齐后的原始低频近似；
- `low`：经过可选 LowSmooth 校准后的低频语义近似；
- `residual`：核心残差 `F - F_l`；
- `high_raw`：HFE 后但 gate 前的高频特征；
- `high_hat`：gate 后用于 decoder skip 和 aux head 的高频特征；
- `gate`：目标感知 gate map。

---

## 3. 改进 1：LowSmooth 轻量低通校准

### 3.1 设计目的

LowSmooth 的目的不是在输入端显式分解高低频，而是对 **Mamba 输出后的 low representation** 做轻量校准，使其更稳定地呈现低频偏好。

重要区别：

- TinyViM / Wave-Mamba / Laplace-Mamba 通常是先用 Laplace、Wavelet、Pooling 等外部算子对输入 `F` 做显式分频，再把低频送入 Mamba。
- MIRFD v2 仍然是先通过 SS2D 得到 `low0 = SS2D(F)`，LowSmooth 只对 Mamba-induced representation 做轻量平滑校准。
- 高频分支仍来自 `residual = F - low`，核心仍是 input-output residual。

### 3.2 推荐实现

```python
class FixedDepthwiseBlur(nn.Module):
    def __init__(self, dim):
        super().__init__()
        kernel = torch.tensor(
            [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
        ) / 16.0
        weight = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.groups = dim

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.groups)
```

```python
class LowSmooth(nn.Module):
    def __init__(self, dim, beta_init=0.3):
        super().__init__()
        self.blur = FixedDepthwiseBlur(dim)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        beta = torch.clamp(self.beta, 0.0, 1.0)
        x_blur = self.blur(x)
        return x + beta * (x_blur - x)
```

配置开关：

```yaml
mirfd:
  use_low_smooth: true
  low_smooth_beta_init: 0.3
```

必须保留消融：

```yaml
use_low_smooth: false
use_low_smooth: true
```

### 3.3 论文表述建议

不要写：

> We decompose the input feature into low-frequency and high-frequency components using smoothing.

推荐写：

> Instead of using a handcrafted frequency decomposition operator to split the input feature before Mamba modeling, we first employ SS2D to produce a Mamba-induced semantic approximation. A lightweight residual smoothing calibration is applied only to the Mamba-induced representation to stabilize its low-frequency preference, while the high-frequency-enriched residual is still derived from the discrepancy between the input feature and the calibrated Mamba output.

中文意思：

> 本方法不是先用手工频率算子对输入特征进行高低频分解，而是先利用 SS2D 生成 Mamba 诱导的语义近似。LowSmooth 仅作为 Mamba 输出后的轻量低频校准，用于稳定其低频偏好；高频分支仍然来自输入特征与校准后 Mamba 输出之间的残差。

---

## 4. 改进 2：High branch 保留 residual 直连

当前：

```python
high = self.hfe(residual)
```

可能把原始 residual 信息洗掉。建议改为：

方案 A，推荐：

```python
hfe_out = self.hfe(residual)
high_raw = self.high_proj(torch.cat([residual, hfe_out], dim=1))
```

```python
self.high_proj = nn.Sequential(
    nn.Conv2d(dim * 2, dim, kernel_size=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

方案 B，简单稳定：

```python
high_raw = residual + self.hfe(residual)
```

配置：

```yaml
mirfd:
  high_residual_mode: concat_proj  # options: concat_proj, add
```

---

## 5. 改进 3：Gate 从抑制型改为增强型

当前：

```python
high_hat = gate * high
```

风险：如果目标区域 gate 值低，会直接压掉小目标高频响应。

v2 推荐：

```python
high_hat = (1.0 + alpha * gate) * high_raw
```

实现：

```python
self.gate_alpha = nn.Parameter(torch.tensor(1.0))

alpha = torch.clamp(self.gate_alpha, 0.0, 2.0)
high_hat = (1.0 + alpha * gate) * high_raw
```

支持三种模式：

```yaml
mirfd:
  gate_mode: enhance  # options: suppress, enhance, half_enhance
```

对应：

```python
if gate_mode == "suppress":
    high_hat = gate * high_raw
elif gate_mode == "enhance":
    high_hat = (1.0 + alpha * gate) * high_raw
elif gate_mode == "half_enhance":
    high_hat = (0.5 + gate) * high_raw
```

默认推荐：

```yaml
gate_mode: enhance
```

---

## 6. 改进 4：Stage 1 增加浅层 high skip

当前 decoder 最后一层可能类似：

```python
d1 = self.dec1(d2, e1, torch.zeros_like(e1))
```

建议新增 Stage 1 high skip：

```python
e1_low = self.stage1_blur(e1)
e1_residual = e1 - e1_low
e1_high = self.stage1_hfe(e1_residual)
d1 = self.dec1(d2, e1, e1_high)
```

新增模块：

```python
self.stage1_blur = FixedDepthwiseBlur(dim_stage1)
self.stage1_hfe = HighFrequencyEnhancer(dim_stage1)
```

配置：

```yaml
model:
  use_stage1_high_skip: true
```

消融：

| Variant | Purpose |
|---|---|
| without stage1 high skip | 当前默认 |
| with stage1 high skip | 验证浅层高频是否改善小目标定位、Pd、nIoU |

---

## 7. 改进 5：Spectral loss 改到 raw residual / high_raw

当前如果 spectral loss 使用 gate 后的 `high_hat`，会把 gate 误差混入频谱约束。

建议：

- decoder skip 使用 `high_hat`
- auxiliary head 使用 `high_hat`
- high spectral loss 使用 `residual` 或 `high_raw`
- 频谱诊断同时输出 `residual`、`high_raw`、`high_hat`

配置：

```yaml
loss:
  spectral_high_target: high_raw  # options: residual, high_raw, high_hat
```

默认推荐：

```yaml
spectral_high_target: high_raw
```

总 loss 可以保持：

```python
loss = loss_seg + lambda_aux * loss_aux + lambda_low * loss_spec_low + lambda_high * loss_spec_high
```

建议初始：

```yaml
spectral_low_weight: 0.001
spectral_high_weight: 0.001
```

IRSTD-1K 先从 0 开始，再逐步加 spectral loss。

---

## 8. 改进 6：诊断输出增强

建议 validation 或单独脚本输出以下统计。

### 8.1 高频能量比例

对以下特征计算：

```text
input_feature F
low
residual
high_raw
high_hat
```

指标：

```python
R_high(X) = sum(abs(FFT(X)) * M_high) / (sum(abs(FFT(X))) + eps)
```

预期：

```text
R_high(residual or high_raw) > R_high(F) > R_high(low)
```

### 8.2 前景/背景响应

对以下特征计算 fg/bg：

```text
low
residual
high_raw
high_hat
gate
```

重点关注：

```text
high_raw fg/bg
high_hat fg/bg
gate fg-bg
```

如果 `gate fg-bg < 0` 仍大量出现，说明 gate 仍在压目标。

### 8.3 可视化

保存：

```text
input image
GT mask
prediction mask
low response
residual response
high_raw response
high_hat response
gate map
FFT(low)
FFT(residual)
FFT(high_raw)
```

---

## 9. 改进 7：训练稳定性

### 9.1 NaN 保护

```python
if not torch.isfinite(loss):
    print(f"[ERROR] non-finite loss at epoch={epoch}, iter={i}, loss={loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    raise FloatingPointError("Non-finite loss detected.")
```

### 9.2 梯度裁剪

配置：

```yaml
train:
  grad_clip_norm: 1.0
```

代码：

```python
if grad_clip_norm is not None and grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
```

### 9.3 checkpoint

建议保存：

```text
best_iou.pt
best_niou.pt
best_dice.pt
best_pd_fa.pt
last_finite.pt
```

避免非有限 loss 后继续保存 `last.pt`。

---

## 10. 推荐实验顺序

### P0：MIRFD v2 基础结构

```yaml
use_low_smooth: true
high_residual_mode: concat_proj
gate_mode: enhance
use_stage1_high_skip: false
use_spectral_loss: false
use_aux_head: false
```

目的：先看结构本身是否比 v1 稳。

### P0：Gate v1 vs Gate v2

```yaml
gate_mode: suppress
gate_mode: enhance
```

看：

```text
IoU
nIoU
Pd
Fa
gate fg-bg
high_hat fg/bg
```

### P1：Stage1 high skip

```yaml
use_stage1_high_skip: true
```

重点看 IRSTD-1K 的 nIoU 和 Pd。

### P1：Auxiliary HF loss

```yaml
use_aux_head: true
aux_weight: 0.2
```

若不稳：

```yaml
aux_weight: 0.1
```

### P2：Spectral loss v2

```yaml
use_spectral_loss: true
spectral_low_weight: 0.001
spectral_high_weight: 0.001
spectral_high_target: high_raw
```

---

## 11. 公平消融矩阵

后续写论文时，必须在相同预处理、优化器、schedule 下重跑：

| Variant | Description |
|---|---|
| Conv baseline | 无 Mamba，无 MIRFD |
| SS2D-only | 只加 SS2D，不做 residual frequency decoupling |
| MIRFD v1 | 当前版本 |
| MIRFD v2 w/o LowSmooth | 去掉 LowSmooth |
| MIRFD v2 w/o Gate | 去掉 gate |
| MIRFD v2 with suppressive gate | 旧 gate |
| MIRFD v2 with enhance gate | 新 gate |
| MIRFD v2 w/o Stage1 high skip | 去掉浅层 high skip |
| MIRFD v2 full | 完整模型 |
| AvgPool residual | `F - Up(AvgPool(F))` |
| Laplace residual | 显式 Laplace high |
| Sobel residual | Sobel edge |
| Pyramid avgpool residual | avgpool 金字塔残差，作为轻量外部分频对照；真正 Haar wavelet 可后续补充 |

最关键对比：

```text
mamba_residual vs avgpool_residual vs laplace_residual vs sobel_residual vs pyramid_avgpool_residual
```

---

## 12. Codex 修改 checklist

请 Codex 按以下顺序修改：

- [ ] 修改 `MIRFDBlock`，返回 `low0 / low / residual / high_raw / high_hat / gate`。
- [ ] 新增 `FixedDepthwiseBlur` 和 `LowSmooth`。
- [ ] high 分支改为 `concat(residual, HFE(residual)) + projection` 或 `residual + HFE(residual)`。
- [ ] gate 支持 `suppress / enhance / half_enhance`。
- [ ] decoder 使用 `high_hat`。
- [ ] spectral loss 使用 `high_raw` 或 `residual`。
- [ ] Stage 1 增加可选 shallow high skip。
- [ ] loss config 增加 `spectral_high_target`。
- [ ] validation 诊断输出增加 `residual / high_raw / high_hat / gate`。
- [ ] train 脚本增加 NaN 保护、梯度裁剪和 checkpoint 管理。
- [ ] config 增加所有 ablation 开关。
- [ ] 增加 `pyramid_avgpool` residual ablation；真正 Haar wavelet high-frequency 如需写入论文需后续单独实现。
- [ ] README 和实验记录文档更新 v2 说明。

---

## 13. 需要避免的误解

### 13.1 LowSmooth 会不会像已有文章？

会有一点表面相似，因为它确实是平滑/低通操作。但它在 MIRFD v2 中的角色不同：

- 它不是输入分频器；
- 它不是先验地把 `F` 拆成 `F_low` 和 `F_high`；
- 它只作用于 SS2D/Mamba 输出后的 `low0`；
- 高频分支仍来自 `F - low`，也就是 Mamba-induced residual。

因此，论文中必须强调：

> LowSmooth is used as a lightweight calibration on the Mamba-induced representation, rather than a handcrafted frequency decomposition operator applied to the input feature.

### 13.2 如何避免削弱创新性？

必须做以下消融：

| Experiment | Purpose |
|---|---|
| MIRFD v2 without LowSmooth | 证明 Mamba-induced residual 本身有效 |
| MIRFD v2 with LowSmooth | 证明校准进一步增强 |
| AvgPool residual | 证明不是普通 smoothing residual |
| Laplace/Sobel/Pyramid residual | 证明不是传统显式分频 |

只要 `without LowSmooth` 已经优于 baseline，而 `with LowSmooth` 进一步提升，就不会严重削弱核心创新。

---

## 14. 最终一句话

MIRFD-Net v2 的核心是：让 SS2D/Mamba 输出更稳定地作为低频语义近似，让 input-output residual 更直接地保留下来，并让 gate 从“可能压制目标”的硬开关变成“目标相关高频增强权重”。这样才能更有力地证明 MIRFD 不是普通高频增强，而是利用 Mamba 低频建模偏好诱导出的红外小目标高频残差恢复机制。
