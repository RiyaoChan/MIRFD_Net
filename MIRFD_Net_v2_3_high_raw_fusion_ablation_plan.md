# MIRFD-Net v2.3 调整计划：绕过 Gate/High_hat，验证 High_raw 与 Residual 主路径

> 目的：根据 v2.2 FSRE 的内部特征统计和可视化结果，验证当前性能瓶颈是否来自 `gate -> high_hat` 调制路径。  
> 核心假设：`residual` 和 `high_raw` 中已经包含较清晰的小目标响应，但 `gate` 没有稳定学到 target-aware selection，导致 `high_hat` 将小目标高频和背景高频混合，污染 MIRFD Block 主路径。  
> 因此，下一步不是继续增强 FSRE，而是让 MIRFD Block 和 Decoder 可以直接使用 `high_raw` 或 `residual`，绕过 `high_hat`。

---

## 1. 背景观察

v2.2 FSRE feature diagnostics 和 20-sample visualization 显示：

1. `stage-1 residual = F - low` 明显关注小目标，支持 Mamba-induced residual 的核心假设。
2. 在 stage-2 和 stage-3 中，`residual` 与 `high_raw` 仍能看到小目标响应。
3. 在不少样本中，`high_raw` 对小目标的响应比 `residual` 更强，说明 FSRE/HFE 对 high branch 并非无效。
4. 但是 `gate` 热力图没有稳定地强调小目标，尤其在 stage-3/stage-4 中，背景区域常常比小目标更强。
5. `high_hat` 经 gate 调制后，常常表现为小目标高频和背景高频混合，整张图高频变亮，小目标不再突出。
6. 当前 Decoder 已可通过 `decoder_high_source: high_raw` 使用 gate 前特征，但 MIRFD Block 内部 fusion 仍固定使用 `high_hat`：

```python
out = self.fuse(torch.cat([low, high_hat], dim=1)) + x
```

因此，即使 decoder 使用 `high_raw`，`high_hat` 仍会通过 block 主路径影响后续 encoder 表示。

---

## 2. 当前 MIRFDBlock 逻辑

当前 forward 大致为：

```python
def forward(self, x: torch.Tensor, return_branches: bool = False):
    low0, low, residual = self._low_and_residual(x)
    high_raw = self._high_branch(residual)
    gate = self.gate(low, residual) if self.gate is not None else torch.ones_like(high_raw)
    high_hat = self._apply_gate(high_raw, gate)

    if self.fusion == "concat":
        out = self.fuse(torch.cat([low, high_hat], dim=1)) + x
    else:
        out = x + low + self.gamma * high_hat

    if return_branches:
        return out, {
            "low0": low0,
            "low": low,
            "high": high_hat,
            "high_raw": high_raw,
            "high_hat": high_hat,
            "residual": residual,
            "gate": gate,
        }
    return out
```

主要问题：

```text
high_hat 是 gate 调制后的结果。
如果 gate 学错，high_hat 会比 high_raw 更差。
当前 block 主路径固定融合 high_hat，可能污染 encoder 表示。
```

---

## 3. 不建议直接 `return high_raw`

不要把 MIRFDBlock 的主输出直接改成：

```python
return high_raw
```

原因：

1. `high_raw` 只是高频残差分支，缺少 low branch 的语义信息；
2. 直接返回 high_raw 会破坏 encoder 主干的信息连续性；
3. 后续 stage 会变成高频堆叠，容易丢失上下文和目标定位信息；
4. 训练稳定性风险较大。

更合理的方式是：

```text
MIRFDBlock 仍然输出 out；
但 out 中 high 分支的 fusion source 从 high_hat 改为可配置：
high_hat | high_raw | residual
```

---

## 4. 新增配置：block_fusion_high_source

请新增配置项：

```yaml
model:
  mirfd:
    block_fusion_high_source: high_raw  # options: high_hat, high_raw, residual
```

含义：

| 选项 | 含义 |
|---|---|
| `high_hat` | 当前默认设计，使用 gate 调制后的 high feature |
| `high_raw` | 使用 gate 前的 enhanced residual，绕过 gate |
| `residual` | 使用原始 Mamba-induced residual，更保守 |

建议默认仍保持兼容：

```yaml
block_fusion_high_source: high_hat
```

但 v2.3 实验优先使用：

```yaml
block_fusion_high_source: high_raw
```

或：

```yaml
block_fusion_high_source: residual
```

---

## 5. MIRFDBlock 修改方案

### 5.1 `__init__` 新增参数

在 `MIRFDBlock.__init__()` 中增加：

```python
block_fusion_high_source: str = "high_hat"
```

并检查合法性：

```python
valid_sources = {"high_hat", "high_raw", "residual"}
if block_fusion_high_source not in valid_sources:
    raise ValueError(
        f"Unsupported block_fusion_high_source: {block_fusion_high_source}. "
        f"Expected one of {valid_sources}."
    )
self.block_fusion_high_source = block_fusion_high_source
```

---

### 5.2 新增 high source 选择函数

建议在 `MIRFDBlock` 中新增内部函数：

```python
def _select_high_for_fusion(
    self,
    residual: torch.Tensor,
    high_raw: torch.Tensor,
    high_hat: torch.Tensor,
) -> torch.Tensor:
    if self.block_fusion_high_source == "high_hat":
        return high_hat
    if self.block_fusion_high_source == "high_raw":
        return high_raw
    if self.block_fusion_high_source == "residual":
        return residual
    raise RuntimeError(f"Invalid block_fusion_high_source: {self.block_fusion_high_source}")
```

---

### 5.3 forward 修改

将原来的：

```python
if self.fusion == "concat":
    out = self.fuse(torch.cat([low, high_hat], dim=1)) + x
else:
    out = x + low + self.gamma * high_hat
```

改成：

```python
high_for_fusion = self._select_high_for_fusion(
    residual=residual,
    high_raw=high_raw,
    high_hat=high_hat,
)

if self.fusion == "concat":
    out = self.fuse(torch.cat([low, high_for_fusion], dim=1)) + x
else:
    out = x + low + self.gamma * high_for_fusion
```

---

### 5.4 return branches 增加字段

建议 return 中增加：

```python
"high_for_fusion": high_for_fusion,
"block_fusion_high_source": self.block_fusion_high_source,
```

保留已有字段：

```python
"high": high_for_fusion,
"high_raw": high_raw,
"high_hat": high_hat,
"residual": residual,
"gate": gate,
```

注意：之前 `"high"` 对应 `high_hat`。v2.3 后建议 `"high"` 对应实际用于 block fusion 的 high feature，即 `high_for_fusion`。  
为了避免兼容问题，也可以新增 `"high_for_fusion"`，保留 `"high": high_hat`。但从诊断角度，推荐让 `"high"` 表示实际使用的 high feature，并显式保留 `"high_hat"`。

推荐：

```python
if return_branches:
    return out, {
        "low0": low0,
        "low": low,
        "high": high_for_fusion,
        "high_for_fusion": high_for_fusion,
        "high_raw": high_raw,
        "high_hat": high_hat,
        "residual": residual,
        "gate": gate,
    }
```

---

## 6. Gate 的处理方式

### 6.1 不建议立即删除 gate 代码

保留 gate 代码，用于 ablation 和诊断。

### 6.2 新增 `gate_mode: none`

如果当前已有 gate mode，可以增加：

```yaml
model:
  mirfd:
    gate_mode: none
```

或：

```yaml
model:
  mirfd:
    use_gate: false
```

推荐统一为：

```yaml
gate_mode: none | enhance | centered | suppress
```

当：

```yaml
gate_mode: none
```

时：

```python
self.gate = None
```

forward 中：

```python
if self.gate is None:
    gate = torch.ones_like(high_raw)
    high_hat = high_raw
else:
    gate = self.gate(low, residual)
    high_hat = self._apply_gate(high_raw, gate)
```

这样可以直接验证：

```text
without gate 是否优于 with gate
```

---

## 7. MIRFDNet / build_model 需要传参

确保 `build_model()` 从 config 读取：

```python
block_fusion_high_source = mirfd_cfg.get("block_fusion_high_source", "high_hat")
```

并传入每个 MIRFD Block。

如果当前 `MIRFDNet` 是通过 `block_kwargs` 创建 stage-2/3/4，请确保：

```python
block_kwargs["block_fusion_high_source"] = mirfd_cfg.get(
    "block_fusion_high_source", "high_hat"
)
```

---

## 8. DecoderBlock 是否需要修改？

一般不需要改 DecoderBlock 结构。

原因：

```text
high_raw、high_hat、residual 的 shape 一致，都是 [B, C, H, W]。
```

只要 `MIRFDNet.forward()` 中已经支持：

```yaml
decoder_high_source: high_raw
```

则 decoder 可以直接接收 high_raw。

需要确认已有逻辑类似：

```python
def _decoder_branch(branch):
    return branch[self.decoder_high_source]
```

然后：

```python
h2 = self._decoder_branch(b2) if 2 in self.high_skip_stages else None
h3 = self._decoder_branch(b3) if 3 in self.high_skip_stages else None
```

v2.3 推荐：

```yaml
decoder_high_source: high_raw
```

或保守版本：

```yaml
decoder_high_source: residual
```

---

## 9. 推荐实验矩阵

### Experiment A：当前 v2.2 对照

```yaml
model:
  decoder:
    use_high_residual_skip: true
    high_skip_stages: [1, 2]
    decoder_high_source: high_raw

  mirfd:
    block_fusion_high_source: high_hat
    gate_mode: enhance
    high_enhancer_type: freq_window
    stage1_high_enhancer_type: identity
```

目的：当前 v2.2 对照。

---

### Experiment B：去掉 gate 主路径，block 与 decoder 都用 high_raw

```yaml
model:
  decoder:
    use_high_residual_skip: true
    high_skip_stages: [1, 2]
    decoder_high_source: high_raw

  mirfd:
    block_fusion_high_source: high_raw
    gate_mode: none
    high_enhancer_type: freq_window
    stage1_high_enhancer_type: identity
```

目的：

```text
验证 high_raw 是否已经比 high_hat 更适合作为主路径特征。
```

这是最推荐优先跑的实验。

---

### Experiment C：保守 residual 主干，decoder 用 high_raw

```yaml
model:
  decoder:
    use_high_residual_skip: true
    high_skip_stages: [1, 2]
    decoder_high_source: high_raw

  mirfd:
    block_fusion_high_source: residual
    gate_mode: none
    high_enhancer_type: freq_window
    stage1_high_enhancer_type: identity
```

目的：

```text
验证 high_raw 是否适合进入 encoder 主路径；
如果 high_raw 仍混入背景，则 block 主干用 residual 可能更稳。
```

---

### Experiment D：block 用 high_raw，但保留 gate 诊断

```yaml
model:
  decoder:
    use_high_residual_skip: true
    high_skip_stages: [1, 2]
    decoder_high_source: high_raw

  mirfd:
    block_fusion_high_source: high_raw
    gate_mode: enhance
    high_enhancer_type: freq_window
    stage1_high_enhancer_type: identity
```

目的：

```text
保留 gate/high_hat 的计算和诊断，但不让 high_hat 进入 block 主路径。
```

如果 D 优于 A，说明问题主要来自 high_hat 融合，而不是 gate 计算本身。

---

## 10. 推荐优先级

优先跑：

```text
Experiment B
Experiment C
```

其中 B 是最关键实验：

```text
block_fusion_high_source = high_raw
decoder_high_source = high_raw
gate_mode = none
```

如果 B 提升，可以说明：

```text
residual/high_raw 中已有小目标特征；
gate/high_hat 是当前主要瓶颈；
MIRFD Block 应采用 direct high_raw residual fusion。
```

如果 C 优于 B，则说明：

```text
high_raw 适合 decoder 浅层补偿，但不适合作为 encoder 主路径；
encoder 主路径应更保守地使用 residual。
```

---

## 11. 诊断脚本需要同步更新

`diagnose_feature_statistics.py` 建议增加或确认以下字段：

```text
high_for_fusion_fg_bg
R_high_high_for_fusion
block_fusion_high_source
```

如果当前 `"high"` 字段已经改为 `high_for_fusion`，则 summary 里需要明确说明：

```text
high = actual high feature used for block fusion
high_raw = before gate
high_hat = after gate
```

建议 raw CSV 增加：

```text
R_high_high_for_fusion
high_for_fusion_fg_bg
```

这样后续可以直接比较：

```text
residual vs high_raw vs high_hat vs high_for_fusion
```

---

## 12. Smoke test 需要新增断言

请在 smoke test 中新增：

1. 构建 `block_fusion_high_source=high_raw` 的模型；
2. forward 时 `return_features=True`；
3. 断言返回 features 中包含：

```python
"high_for_fusion"
"high_raw"
"high_hat"
"residual"
"gate"
```

4. 当 `block_fusion_high_source=high_raw` 时，检查：

```python
torch.allclose(high_for_fusion, high_raw, atol=1e-6)
```

5. 当 `gate_mode=none` 时，检查：

```python
torch.allclose(high_hat, high_raw, atol=1e-6)
```

6. 当 `block_fusion_high_source=residual` 时，检查：

```python
torch.allclose(high_for_fusion, residual, atol=1e-6)
```

---

## 13. 论文解释建议

如果 Experiment B 或 C 提升，可以写成：

```text
Although the Mamba-induced residual and FSRE-enhanced high_raw features preserve clear small-target responses, the learned gate does not consistently produce target-aware modulation. In deeper stages, the gate tends to mix target-related residuals with clutter-like background high-frequency responses. Therefore, MIRFD decouples residual enhancement from residual selection and directly fuses high_raw/residual cues instead of relying on gated high_hat features.
```

中文：

```text
尽管 Mamba 诱导残差和 FSRE 增强后的 high_raw 中保留了清晰的小目标响应，但当前 gate 不能稳定地产生目标感知调制。在较深层中，gate 容易将目标相关残差与背景杂波高频混合。因此，MIRFD 将残差增强与残差选择解耦，直接融合 high_raw 或 residual，而不是依赖 gate 调制后的 high_hat。
```

---

## 14. 最终目标

v2.3 的目标不是提出一个更复杂的新模块，而是验证一个关键机制：

```text
小目标信息已经存在于 residual/high_raw；
错误的 gate/high_hat 调制可能是当前性能瓶颈；
直接 high_raw/residual fusion 是否优于 gated high_hat fusion。
```

如果该实验成立，MIRFD 的主线将更清晰：

```text
SS2D/Mamba low semantic approximation
→ Mamba-induced residual
→ FSRE high_raw enhancement
→ direct high_raw/residual fusion
→ shallow decoder high skip
```
