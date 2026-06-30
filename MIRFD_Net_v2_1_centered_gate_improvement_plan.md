# MIRFD-Net v2.1 改进计划：Centered Gate 与 High Skip 选择性增强

> 目的：根据 v2 feature diagnostics 的可视化结果，进一步修改 MIRFD-Net，使 high branch 从“全图高频增强”转向“目标相关高频选择性增强”。  
> 当前问题：`residual/high_raw` 已经包含小目标信息，但 `high_hat` 经过 enhance gate 后过强、过宽泛，背景纹理、云层边缘、地平线、建筑边缘等也被增强并送入 decoder。  
> 适用任务：红外小目标分割，数据集包括 NUAA-SIRST、NUDT-SIRST、IRSTD-1K。

---

## 1. 当前诊断结论

根据 `docs/visualizations/v2_feature_diagnostics` 中的结构诊断图和 FFT 诊断图，当前 v2 的主要问题不是 high branch 没有学到高频，而是 high branch 过强、不够选择性。

### 1.1 low0 与 low 差异不大

当前 `LowSmooth` 对 `low0` 的改变较小，`low0` 和 `low` 在空间响应上非常接近。说明 LowSmooth 只是轻量平滑校准，不是主要矛盾。

建议：

- 保留 `use_low_smooth` 作为开关；
- 继续做 `with / without LowSmooth` 消融；
- 不要把 LowSmooth 作为论文核心创新点。

---

### 1.2 residual 确实包含目标信息，但不是 target-specific residual

公式：

```text
residual = F - low
```

可视化显示，`residual` 中确实出现了小目标响应，但同时也包含大量背景高频信息，例如海面纹理、云层边界、地平线、建筑边缘、树枝和地物结构、大范围亮暗变化。

因此，更准确的表述应该是：

```text
residual is a high-frequency-enriched residual, but not a target-specific residual.
```

也就是说，`F - low` 支持 MIRFD 的核心 idea，但还需要后续目标选择机制，否则会增强背景杂波。

---

### 1.3 high_raw 相对 high_hat 更干净

当前结构大致为：

```text
residual = F - low
high_raw = HFE(residual)
gate = TargetAwareGate(low, residual)
high_hat = (1 + alpha * gate) * high_raw
```

可视化显示，很多样本中 `high_raw` 比 `high_hat` 更稀疏、更干净；而 `high_hat` 出现了大面积背景增强。

这说明问题主要不在 residual 或 HFE，而是在：

```text
high_hat = (1 + alpha * gate) * high_raw
```

当前 enhance gate 的最小缩放系数是 1，只能增强，不能抑制背景。因此只要 gate 对背景有响应，背景高频也会被一起放大。

---

### 1.4 deep high skip 可能把粗粒度背景残差送回 decoder

Stage-3 / Stage-4 的 `high_hat` 经常变成大块背景区域响应，而不是局部小目标细节。深层 high skip 的空间分辨率较低，容易把 coarse background residual 注入 decoder。

建议：

- 默认只保留 Stage-1 / Stage-2 high skip；
- Stage-3 / Stage-4 high skip 作为消融；
- 避免所有尺度都向 decoder 注入 high residual。

---

## 2. 改进目标

MIRFD-Net v2.1 的核心目标：

1. 让 gate 既能增强目标，也能抑制背景；
2. 限制 high skip 层级，减少深层粗背景残差回流 decoder；
3. 保留 residual 的原始含义，避免 HFE/proj 过度重构；
4. 新增诊断指标，区分 residual、high_raw、gate、high_hat 的作用。

---

## 3. 改进一：新增 Centered Gate

### 3.1 当前 enhance gate 的问题

当前 gate 模式为：

```python
high_hat = (1.0 + alpha * gate) * high_raw
```

其中：

```text
gate ∈ [0, 1]
```

如果 `alpha = 1`，缩放范围为：

```text
scale ∈ [1, 2]
```

这意味着：

- gate 接近 1 的位置：高频增强 2 倍；
- gate 接近 0 的位置：高频保持 1 倍；
- 没有任何位置会被抑制。

因此，当前 enhance gate 只能做“增强”，不能做“选择”。如果 gate 图不够 target-aware，而是对背景纹理、地平线或大块区域也有响应，就会把背景高频一起增强。

---

### 3.2 Centered Gate 的公式

建议新增：

```python
high_hat = (1.0 + alpha * (gate - 0.5)) * high_raw
```

其中：

```text
gate ∈ [0, 1]
```

当 `alpha = 1` 时：

```text
scale = 1.0 + (gate - 0.5)
scale ∈ [0.5, 1.5]
```

即：

- gate > 0.5：增强 high_raw；
- gate = 0.5：保持 high_raw；
- gate < 0.5：抑制 high_raw。

这就是 centered gate 能同时“增强目标”和“抑制背景”的原因。

---

### 3.3 直观解释

Centered gate 把 `0.5` 作为中性点：

```text
gate = 0.5  -> 不改变 high_raw
gate > 0.5  -> 认为该位置更像目标相关残差，因此增强
gate < 0.5  -> 认为该位置更像背景杂波，因此抑制
```

因此它比当前 enhance gate 更适合当前问题。

当前 enhance gate：

```text
scale = 1 + alpha * gate
```

只有增强作用：

```text
gate = 0    -> scale = 1
gate = 1    -> scale = 2
```

Centered gate：

```text
scale = 1 + alpha * (gate - 0.5)
```

具有双向调制作用：

```text
gate = 0    -> scale = 0.5
gate = 0.5  -> scale = 1.0
gate = 1    -> scale = 1.5
```

这使 gate 从“高频放大器”变成“高频选择器”。

---

### 3.4 建议实现

在 `MIRFDBlock._apply_gate()` 中新增模式：

```python
elif self.gate_mode == "centered":
    alpha = torch.clamp(self.gate_alpha, 0.0, 2.0)
    scale = 1.0 + alpha * (gate - 0.5)
    scale = torch.clamp(scale, 0.25, 1.75)
    high_hat = scale * high_raw
```

建议支持配置：

```yaml
mirfd:
  gate_mode: centered
  gate_alpha_init: 1.0
  gate_scale_min: 0.25
  gate_scale_max: 1.75
```

如果当前代码暂时不支持 `gate_scale_min/max`，可以先固定在代码里：

```python
scale = torch.clamp(scale, 0.25, 1.75)
```

---

### 3.5 保留现有 gate 模式做消融

继续保留：

```yaml
gate_mode: suppress       # high_hat = gate * high_raw
gate_mode: enhance        # high_hat = (1 + alpha * gate) * high_raw
gate_mode: half_enhance   # high_hat = (0.5 + gate) * high_raw
gate_mode: centered       # high_hat = (1 + alpha * (gate - 0.5)) * high_raw
```

推荐主实验先用：

```yaml
gate_mode: centered
```

---

## 4. 改进二：限制 high residual skip 的层级

### 4.1 当前问题

当前 decoder 可能使用：

```text
Stage-1 high skip
Stage-2 high skip
Stage-3 high skip
Stage-4 high skip
```

但可视化显示：

- Stage-1 / Stage-2 high 更接近小目标局部细节；
- Stage-3 / Stage-4 high 经常变成大块背景区域响应；
- 深层 high skip 可能把 coarse background residual 送回 decoder。

---

### 4.2 建议新增 high_skip_stages 配置

新增配置：

```yaml
model:
  high_skip_stages: [1, 2]
```

含义：

```text
1 -> 使用 Stage-1 shallow high skip
2 -> 使用 Stage-2 MIRFD high_hat
3 -> 使用 Stage-3 MIRFD high_hat
4 -> 使用 Stage-4 MIRFD high_hat
```

推荐默认：

```yaml
high_skip_stages: [1, 2]
```

作为消融：

```yaml
high_skip_stages: [1, 2, 3, 4]
high_skip_stages: [2, 3, 4]
high_skip_stages: [1, 2]
high_skip_stages: [2]
high_skip_stages: []
```

---

### 4.3 代码逻辑建议

在 `MIRFDNet.__init__` 中：

```python
self.high_skip_stages = set(high_skip_stages or [])
```

forward 中：

```python
h4 = b4["high_hat"] if (b4 is not None and 4 in self.high_skip_stages) else None
h3 = b3["high_hat"] if (b3 is not None and 3 in self.high_skip_stages) else None
h2 = b2["high_hat"] if (b2 is not None and 2 in self.high_skip_stages) else None

if 1 in self.high_skip_stages:
    e1_low = self.stage1_blur(e1)
    e1_residual = e1 - e1_low
    h1 = self.stage1_hfe(e1_residual)
else:
    h1 = None

d3 = self.dec3(e4, e3, h3)
d2 = self.dec2(d3, e2, h2)
d1 = self.dec1(d2, e1, h1)
```

注意：如果 decoder stage 的 high input 为 None，`DecoderBlock` 应该内部用 zeros 或跳过该分支，保持维度一致。

---

## 5. 改进三：High branch 保留 residual 原义

### 5.1 residual 和 high_raw 的区别

#### residual

公式：

```text
residual = F - low
```

含义：

```text
输入特征 F 与 Mamba-induced low representation 之间的差异
```

它是 MIRFD 的核心概念，代表被 Mamba/SS2D low branch 弱化的局部变化，也包括小目标边缘和亮点、背景纹理和边缘、所有未被 low branch 表达好的信息。

因此，residual 是最原始的 input-output difference。

#### high_raw

公式：

```text
high_raw = HFE(residual)
```

或：

```text
high_raw = Proj([residual, HFE(residual)])
```

含义：

```text
经过高频增强模块处理后的 residual 表示
```

它不是原始残差，而是网络学习后的高频候选特征。它可能更适合 decoder 使用，但也可能引入重构、放大或混合。

区别总结：

| 特征 | 来源 | 是否原始残差 | 作用 |
|---|---|---|---|
| residual | `F - low` | 是 | 证明 Mamba-induced residual 的核心假设 |
| high_raw | `HFE(residual)` 或 `Proj([residual, HFE(residual)])` | 否 | 对 residual 做局部增强和通道融合，供 gate/decoder 使用 |
| high_hat | `Gate(high_raw)` | 否 | gate 调制后的高频 skip，最终送入 decoder |

因此：

- 论文中证明“残差富集高频”时，优先分析 `residual`；
- 训练 spectral high loss 时，可以约束 `high_raw` 或 `residual`；
- decoder 使用 `high_hat`；
- 可视化必须同时展示 `residual / high_raw / gate / high_hat`。

---

### 5.2 HFE 是什么

HFE = High-Frequency Enhancer，高频增强模块。

它的作用是对 `residual` 进行局部高频建模和增强。一般结构类似：

```text
residual
  -> DWConv 3×3
  -> DWConv 5×5
  -> concat
  -> PWConv 1×1
  -> BN + activation
```

或者：

```text
HFE(residual) = PWConv([DWConv3x3(residual), DWConv5x5(residual)])
```

其中：

- `DWConv 3×3`：捕获小范围边缘、亮点、局部突变；
- `DWConv 5×5`：捕获稍大一点的局部背景纹理和目标邻域；
- `PWConv 1×1`：做通道融合；
- BN/activation：增强表达能力。

---

### 5.3 HFE 的作用

HFE 不是为了重新定义高低频，而是为了把原始残差变成更适合分割 decoder 使用的高频候选特征。

它主要有三个作用：

1. **增强小目标局部结构**  
   小目标可能只有几个像素，直接用 residual 可能响应弱，HFE 可以增强局部亮点和边缘。

2. **整合多尺度局部信息**  
   3×3 和 5×5 depthwise conv 可以同时关注目标中心、边缘和邻域背景。

3. **通道重整与降噪**  
   residual 是直接相减得到的，可能分布杂乱；HFE 可以学习过滤部分无用通道。

但 HFE 也有风险：

> 如果 HFE 太强，它可能把 residual 变成普通卷积分支，削弱 `F - low` 的核心含义，并放大背景纹理。

---

### 5.4 建议新增 hfe_scale

为了避免 HFE 过度重构 residual，建议支持：

```python
high_raw = residual + gamma * self.hfe(residual)
```

其中：

```python
self.hfe_scale = nn.Parameter(torch.tensor(hfe_scale_init))
```

推荐：

```yaml
mirfd:
  high_residual_mode: add_scaled
  hfe_scale_init: 0.1
```

实现：

```python
elif self.high_residual_mode == "add_scaled":
    gamma = torch.clamp(self.hfe_scale, 0.0, 1.0)
    high_raw = residual + gamma * self.hfe(residual)
```

继续保留：

```yaml
high_residual_mode: concat_proj
high_residual_mode: add
high_residual_mode: add_scaled
```

推荐主实验优先试：

```yaml
high_residual_mode: add_scaled
hfe_scale_init: 0.1
```

理由：保留 residual 原义，同时允许 HFE 做轻量增强。

---

## 6. 改进四：Gate 辅助约束，可选

如果 centered gate 后 gate 仍然不够 target-aware，可以加入轻量 gate auxiliary loss。

### 6.1 Gate map 生成

对 gate 做通道平均：

```python
gate_map = gate.mean(dim=1, keepdim=True)
```

将 GT mask 下采样到对应 stage：

```python
target_s = F.interpolate(mask, size=gate_map.shape[-2:], mode="nearest")
```

### 6.2 Gate BCE loss

如果 gate 已经经过 sigmoid，直接用 BCE：

```python
loss_gate = F.binary_cross_entropy(gate_map, target_s)
```

推荐权重很小：

```yaml
loss:
  gate_aux_weight: 0.02
```

或：

```yaml
gate_aux_weight: 0.05
```

注意：gate auxiliary loss 不应太强，否则 gate 会退化成 mask predictor，影响特征调制。

---

### 6.3 背景抑制 loss，可选

也可以只惩罚背景区域 gate 偏高：

```python
bg = 1 - target_s
loss_gate_bg = (gate_map * bg).sum() / (bg.sum() + eps)
```

推荐：

```yaml
loss:
  gate_bg_weight: 0.01
```

这比直接 BCE 更温和，目标是减少 gate 全图发亮。

---

## 7. 改进五：继续增强诊断输出

当前诊断已经很好。建议下一版增加数值统计 CSV。

每个样本、每个 stage 输出：

```text
sample_id
stage
R_high_low
R_high_residual
R_high_high_raw
R_high_high_hat
gate_fg_mean
gate_bg_mean
gate_fg_minus_bg
high_raw_fg_bg
high_hat_fg_bg
```

重点判断：

```text
R_high(residual) > R_high(low)
R_high(high_raw) > R_high(low)
gate_fg_mean > gate_bg_mean
high_hat_fg_bg >= high_raw_fg_bg
```

如果 `high_hat_fg_bg < high_raw_fg_bg`，说明 gate 反而削弱了目标选择性。

---

## 8. 推荐下一轮实验

### Experiment A：Centered Gate

```yaml
mirfd:
  gate_mode: centered
  gate_alpha_init: 1.0
  high_residual_mode: concat_proj
model:
  high_skip_stages: [1, 2, 3, 4]
loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：只验证 centered gate 是否改善 high_hat 过强问题。

---

### Experiment B：Shallow High Skip

```yaml
mirfd:
  gate_mode: enhance
  high_residual_mode: concat_proj
model:
  high_skip_stages: [1, 2]
loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：只验证关闭 S3/S4 high skip 是否减少背景污染。

---

### Experiment C：Centered Gate + Shallow High Skip

```yaml
mirfd:
  gate_mode: centered
  gate_alpha_init: 1.0
  high_residual_mode: concat_proj
model:
  high_skip_stages: [1, 2]
loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：同时解决 gate 和 deep high skip 问题。当前最推荐。

---

### Experiment D：Centered Gate + Shallow High Skip + add_scaled HFE

```yaml
mirfd:
  gate_mode: centered
  gate_alpha_init: 1.0
  high_residual_mode: add_scaled
  hfe_scale_init: 0.1
model:
  high_skip_stages: [1, 2]
loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：保留 residual 原义，降低 HFE 重构强度。当前第二推荐。

---

### Experiment E：Gate auxiliary loss

仅在 C 或 D 有改善但 gate 仍不够 target-aware 时启用：

```yaml
loss:
  gate_aux_weight: 0.02
```

或：

```yaml
loss:
  gate_bg_weight: 0.01
```

---

## 9. Codex 修改 checklist

请 Codex 完成以下修改：

- [ ] 在 `MIRFDBlock._apply_gate()` 中新增 `gate_mode="centered"`。
- [ ] 支持 `gate_scale_min` 和 `gate_scale_max`，或在 centered 模式中固定 clamp 到 `[0.25, 1.75]`。
- [ ] 在 config 中新增 centered gate 配置。
- [ ] 在 `MIRFDNet` 中新增 `high_skip_stages` 配置，支持 `[1,2]`、`[1,2,3,4]` 等。
- [ ] 保证 decoder 在某个 high skip 不启用时仍能正常 forward。
- [ ] 新增 `high_residual_mode="add_scaled"`。
- [ ] 新增 `hfe_scale_init`。
- [ ] 可选：新增 gate auxiliary loss 或 gate background suppression loss。
- [ ] 可选：诊断脚本输出 CSV 数值统计。
- [ ] 新增实验配置：
  - `v2_centered_gate`
  - `v2_shallow_high_skip`
  - `v2_centered_shallow`
  - `v2_centered_shallow_add_scaled`
- [ ] 更新 README 中 MIRFD Block 公式。
- [ ] 更新 EXPERIMENT_RESULTS_AND_ANALYSIS.md，说明 v2 feature diagnostics 的结论。

---

## 10. README 中建议更新的核心公式

推荐写成：

```text
low0      = Align(SS2D(Norm(F)))
low       = LowSmooth(low0)
residual  = F - low
high_raw  = HFE(residual) or residual + gamma · HFE(residual)
gate      = TargetAwareGate(low, residual)
high_hat  = GateModulation(gate, high_raw)
F_out     = Fuse(low, high_hat) + F
```

其中 centered gate 为：

```text
GateModulation(gate, high_raw) = [1 + alpha · (gate - 0.5)] · high_raw
```

---

## 11. 最终结论

当前 v2 diagnostics 表明：

```text
residual/high_raw 方向是对的；
high_hat 的 gate 调制过强、过宽泛；
深层 high skip 可能把 coarse background residual 注入 decoder。
```

因此下一步不应继续简单增强 high branch，而应该让 high branch 更 selective：

```text
centered gate 负责选择性增强/抑制；
high_skip_stages 控制哪些尺度的 high residual 进入 decoder；
add_scaled HFE 保留 residual 的原始含义。
```

最终目标：

> 让 MIRFD 从“全图高频增强”变成“目标相关高频选择性补偿”，从而减少背景杂波、提升 Pd/nIoU，并使论文中的 Mamba-induced high-frequency residual 论点更稳。
