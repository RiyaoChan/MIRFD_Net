# MIRFD-Net v2.5 改进计划：从“高频增强”转向“Context–Residual–Reference 协同”

## 0. 本轮改进的核心目的

当前 v2.2/v2.3/v2.4 的实验已经说明：

1. `residual = F - low` 不是严格高频分量，而是 **Mamba-induced residual / context-discrepancy feature**。
2. Mamba/SS2D 输出 `low` 不是纯低频，其中仍包含部分高频和目标响应。
3. 继续设计更强的 high enhancer（Conv-HFE、FSRE、FFC）并不能稳定提升三个数据集。
4. 当前最稳定的结构是：
   - `block_fusion_high_source = residual`
   - `gate_mode = none`
   - `decoder_high_source = high_raw`
   - `stage1_high_enhancer_type = identity`
   - `high_skip_stages = [1, 2]`
5. 下一步应从“如何增强高频”转为“如何让 Mamba context 指导 residual 中的目标差异选择”。

因此，v2.5 的目标不是继续增强 high_raw，而是回答两个问题：

```text
1. low / Mamba context 是否对红外小目标分割有不可替代作用？
2. residual 中的小目标差异如何被 low context 和 shallow reference 更稳定地筛选？
```

建议将论文主线从：

```text
Mamba-induced residual frequency decoupling
```

调整为：

```text
Mamba-guided context-residual decoupling
```

或者：

```text
Context-guided Mamba residual decoupling for infrared small target segmentation
```

---

## 1. 概念重定义

### 1.1 不再把 `low` 称为纯低频

当前代码中：

```python
fm = self.mamba(self.norm(x))
low0 = self.align(fm)
low = self.low_smooth(low0)
residual = x - low
```

这里的 `low` 更准确应称为：

```text
Mamba-induced context representation
```

或：

```text
Mamba-induced semantic approximation
```

它具有低频/全局背景偏好，但不是纯低频。

### 1.2 不再把 `residual` 称为全部高频

`residual = x - low` 更准确应称为：

```text
Mamba-induced residual
```

或：

```text
context-discrepancy residual
```

它表示输入特征中没有被 Mamba context 充分表达的局部差异。它富含小目标线索，但也混有背景纹理、边缘和建模偏差。

### 1.3 高频不是最终目的

高频的作用是提供：

```text
小目标定位
局部突变
边界细节
局部对比度
```

低频/Mamba context 的作用是提供：

```text
背景结构
全局上下文
目标-背景关系
虚警抑制
```

因此后续模型不应只做 high enhancement，而应做：

```text
context-guided residual selection
```

---

## 2. 从 T-PMambaSR 得到的启发

T-PMambaSR 有两个启发点：

### 2.1 Progressive receptive field

T-PMambaSR 使用：

```text
Window MHSA → Window Scan Mamba → Global Scan Mamba
```

建立从 local、regional 到 global 的平滑感受野过渡。这个思想说明，不应该让所有层都突然直接进入全局扫描。对于红外小目标检测，浅层更需要局部目标细节，深层才更需要全局背景上下文。

### 2.2 Reference-guided high-frequency recovery

T-PMambaSR 的 AHFRM 不是只从处理后的特征中恢复高频，而是同时使用：

```text
Xori: 未处理 / 原始特征，保留较完整高频
Xlf: Transformer/Mamba 处理后的低频偏置特征
```

然后用 `Xori` 中保留的高频作为 reference 去指导恢复 `Xlf` 中退化的高频。

这对 MIRFD 的启发是：

```text
不要只依赖 F - Mamba(F)；
应该保留 shallow / unprocessed reference feature；
用 Mamba context + shallow reference 共同指导 residual 选择。
```

---

## 3. v2.5 第一阶段：先做 Branch Role Diagnostics

在继续改模型前，先回答：

```text
low-only 能不能分割小目标？
residual-only 能不能分割小目标？
low + residual 是否优于单独使用？
low + high_raw 是否真的更好？
```

### 3.1 新增脚本：`scripts/train_branch_probe.py`

建议新增一个轻量 branch probe 训练脚本。它不改变主模型，只用于诊断各分支的分割能力。

使用方式示例：

```bash
python scripts/train_branch_probe.py \
  --config configs/mirfd_nuaa_sirst_ss2d_v2_3_block_residual_gate_none.yaml \
  --checkpoint runs/nuaa_v2_3_block_residual_gate_none/best_iou.pt \
  --dataset-name NUAA-SIRST \
  --branch low,residual,high_raw,low_residual,low_high_raw \
  --stage 2 \
  --epochs 30 \
  --output-csv docs/diagnostics/branch_probe/nuaa_stage2_probe.csv
```

### 3.2 诊断方法

加载训练好的 MIRFDNet，冻结主模型参数：

```python
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
```

对指定 branch feature 训练一个很轻量的 segmentation probe：

```python
probe = nn.Sequential(
    nn.Conv2d(C_in, C_in // 2, 3, padding=1),
    nn.BatchNorm2d(C_in // 2),
    nn.GELU(),
    nn.Conv2d(C_in // 2, 1, 1),
)
```

对不同 branch 输入：

```text
low
residual
high_raw
low + residual: concat([low, residual])
low + high_raw: concat([low, high_raw])
low + residual + high_raw
```

训练 20–30 epoch 即可，不追求最高性能，只用于比较分支潜力。

### 3.3 输出指标

CSV 至少包含：

```text
dataset
stage
branch
probe_iou
probe_niou
probe_dice
probe_pd
probe_fa
false_alarm_rate
```

### 3.4 预期判断

如果：

```text
low + residual > residual-only > low-only
```

说明低频 context 和 residual 是互补的。

如果：

```text
residual-only ≈ low + residual
```

说明当前 low context 没有被有效利用。

如果：

```text
low-only 也有较好 IoU / Pd
```

说明 Mamba context 中仍包含小目标信息，不能把它简单当成背景低频。

如果：

```text
low + high_raw < low + residual
```

说明 high enhancer 仍然引入了不稳定背景高频。

---

## 4. v2.5 第二阶段：Context-Guided Residual Selector（CGRS）

如果 branch probe 证明 `low + residual` 有互补性，则新增一个 context-guided residual selector。

注意：这不是旧的 gate。旧 gate 的问题是没有明确 target-aware，容易学习“哪里高频强就开门”。新的 selector 需要满足：

```text
1. 以 residual 为主，不破坏 residual；
2. low context 只作为背景约束；
3. 初始状态接近 identity；
4. 可以加弱监督，让 selector 更接近目标区域。
```

---

## 5. 新模块：`ContextGuidedResidualSelector`

建议新增文件：

```text
mirfd/models/context_residual.py
```

新增类：

```python
class ContextGuidedResidualSelector(nn.Module):
    def __init__(
        self,
        dim: int,
        use_reference: bool = False,
        gamma_init: float = 0.1,
        norm: str = "batch",
    ):
        ...
```

### 5.1 输入

```text
low: Mamba-induced context representation
residual: Mamba-induced residual
reference: optional shallow reference feature
```

### 5.2 输出

```text
selector: [B, C or 1, H, W]
selected_residual: residual + gamma * (selector - 0.5) * residual
```

### 5.3 为什么 residual-style？

不要直接：

```python
selected = selector * residual
```

因为这样训练初期如果 selector 不准，会直接压掉小目标。

推荐：

```python
selected = residual + gamma * (selector - 0.5) * residual
```

其中：

```python
gamma = nn.Parameter(torch.tensor(0.1))
```

这样初始阶段基本等价于 residual，训练后逐步学习增强或抑制。

### 5.4 伪代码

```python
class ContextGuidedResidualSelector(nn.Module):
    def __init__(self, dim, use_reference=False, gamma_init=0.1, norm="batch"):
        super().__init__()
        self.use_reference = use_reference
        in_dim = dim * (3 if use_reference else 2)

        self.selector = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=1, bias=False),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, low, residual, reference=None):
        if self.use_reference and reference is not None:
            x = torch.cat([low, residual, reference], dim=1)
        else:
            x = torch.cat([low, residual], dim=1)

        s = self.selector(x)
        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        selected = residual + gamma * (s - 0.5) * residual
        return selected, s
```

---

## 6. 接入 MIRFDBlock

### 6.1 新增配置

```yaml
mirfd:
  use_context_residual_selector: true
  selector_stages: [2]
  selector_gamma_init: 0.1
  selector_use_reference: false
  selector_supervision_weight: 0.0
```

### 6.2 修改 MIRFDBlock

在 `MIRFDBlock.__init__()` 中增加：

```python
use_context_residual_selector: bool = False
selector_gamma_init: float = 0.1
selector_use_reference: bool = False
```

构建：

```python
self.residual_selector = (
    ContextGuidedResidualSelector(
        dim,
        use_reference=selector_use_reference,
        gamma_init=selector_gamma_init,
        norm=norm,
    )
    if use_context_residual_selector
    else None
)
```

forward 中：

```python
low0, low, residual = self._low_and_residual(x)
high_raw = self._high_branch(residual)

if self.residual_selector is not None:
    selected_residual, selector_map = self.residual_selector(low, residual, reference=None)
else:
    selected_residual, selector_map = residual, None

# high_for_fusion 来源增加 selected_residual
if block_fusion_high_source == "selected_residual":
    high_for_fusion = selected_residual
```

返回 branches 时增加：

```python
"selected_residual": selected_residual,
"selector": selector_map if selector_map is not None else torch.ones_like(residual),
```

### 6.3 支持新的 block fusion source

当前支持：

```text
high_hat
high_raw
residual
```

新增：

```text
selected_residual
```

---

## 7. Selector 弱监督，可选但建议做 stage-2

旧 gate 失败的核心原因是没有 target-aware 约束。因此 selector 建议支持弱监督。

### 7.1 新增 loss

在 loss 中增加：

```yaml
loss:
  selector_supervision_weight: 0.02
  selector_supervision_stages: [2]
  selector_target_dilate: 1
```

### 7.2 监督目标

把 GT mask resize 到当前 stage 尺寸。由于小目标很小，建议先做 dilation 再 downsample：

```python
gt_dilated = dilate(gt, kernel_size=3)
target = F.interpolate(gt_dilated.float(), size=selector.shape[-2:], mode="area")
target = (target > 0).float()
```

selector map 取通道平均：

```python
selector_map = selector.mean(dim=1, keepdim=True)
```

loss：

```python
L_selector = BCE(selector_map, target)
```

注意：第一轮可以先不加 selector supervision，跑无监督 selector；第二轮再加 `0.02` 权重。

---

## 8. v2.5 第三阶段：Reference-guided Residual Recovery（可选）

借鉴 T-PMambaSR 的 AHFRM，新增 reference 分支。

### 8.1 设计目的

当前 `residual = F - low` 不是完整高频。如果 Mamba 输出仍包含高频，或者 residual 混有背景高频，仅靠 residual 不够。

可以增加：

```text
shallow reference feature
```

用于提供未被 Mamba 平滑/全局化处理过的局部目标线索。

### 8.2 Reference 来源

优先使用：

```text
stage-1 feature / e1
```

因为已有统计证明 stage-1 residual 最目标相关。

对于 stage-2 block，可以将 stage-1 reference 上采样/下采样到 stage-2 尺寸：

```python
ref = F.interpolate(stage1_ref, size=residual.shape[-2:], mode="bilinear", align_corners=False)
ref = ref_proj(ref)
```

### 8.3 Reference HFM

可以实现一个简单 HFM：

```python
class HighFrequencyFilteringModule(nn.Module):
    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x - low
```

reference high:

```python
H_ref = HFM(stage1_ref)
H_deg = HFM(low)
```

融合：

```python
ref_residual = Conv1x1(Concat(residual, H_ref, H_deg))
```

再输入 CGRS：

```python
selected_residual, selector = CGRS(low, residual, reference=ref_residual)
```

### 8.4 不要第一轮直接上 reference

Reference-guided 版本会增加复杂度。建议顺序：

```text
先做 branch probe
再做 CGRS without reference
最后做 CGRS + shallow reference
```

---

## 9. 推荐实验矩阵

### 9.1 Phase A：Branch probe，不改模型

对 v2.3 residual_gate_none 最优 checkpoint 做：

```text
stage-1: low / residual / low+residual
stage-2: low / residual / high_raw / low+residual / low+high_raw
stage-3: low / residual / high_raw / low+residual
```

三个数据集都跑。目标是回答：

```text
低频是否有独立分割能力？
low + residual 是否互补？
high_raw 是否真的比 residual 更适合解码？
```

---

### 9.2 Phase B：CGRS 无监督版本

```yaml
model:
  high_skip_stages: [1, 2]
  decoder_high_source: high_raw
  stage1_high_enhancer_type: identity

mirfd:
  block_fusion_high_source: selected_residual
  use_context_residual_selector: true
  selector_stages: [2]
  selector_gamma_init: 0.1
  selector_use_reference: false
  gate_mode: none
  high_enhancer_type: identity

loss:
  selector_supervision_weight: 0.0
  use_aux_heads: false
```

解释：

```text
只验证 low context 是否能无监督地帮助 residual selection。
先关闭 high enhancer，避免混入 FSRE/FFC 干扰。
```

---

### 9.3 Phase C：CGRS + 弱监督

```yaml
mirfd:
  block_fusion_high_source: selected_residual
  use_context_residual_selector: true
  selector_stages: [2]
  selector_gamma_init: 0.1
  gate_mode: none
  high_enhancer_type: identity

loss:
  selector_supervision_weight: 0.02
  selector_supervision_stages: [2]
  selector_target_dilate: 1
  use_aux_heads: false
```

解释：

```text
解决旧 gate 没有 target-aware 约束的问题。
```

---

### 9.4 Phase D：CGRS + stage-1 reference

```yaml
mirfd:
  block_fusion_high_source: selected_residual
  use_context_residual_selector: true
  selector_stages: [2]
  selector_use_reference: true
  reference_source: stage1
  reference_hfm: avgpool
  gate_mode: none
  high_enhancer_type: identity

loss:
  selector_supervision_weight: 0.02
  use_aux_heads: false
```

解释：

```text
借鉴 AHFRM，用浅层未处理/少处理的 reference high-frequency 辅助 residual selection。
```

---

## 10. 需要新增/修改的文件

### 新增

```text
mirfd/models/context_residual.py
scripts/train_branch_probe.py
configs/v2_5/
```

### 修改

```text
mirfd/models/mirfd_block.py
mirfd/models/mirfd_net.py
mirfd/losses.py
scripts/diagnose_feature_statistics.py
scripts/visualize_features.py
EXPERIMENT_RESULTS_AND_ANALYSIS.md
```

---

## 11. 诊断脚本需要扩展

`diagnose_feature_statistics.py` 增加字段：

```text
R_high_selected_residual
selected_residual_fg_bg
selector_fg_bg
selector_fg_minus_bg
low_fg_bg
low_residual_delta
```

其中：

```text
low_residual_delta = residual_fg_bg - low_fg_bg
```

如果 selector 有效，应该看到：

```text
selected_residual_fg_bg > residual_fg_bg
selector_fg_minus_bg > 0
false_alarm_rate 下降
```

---

## 12. 可视化需要扩展

每个 stage 可视化：

```text
input
GT
prediction
low
residual
selected_residual
selector
high_raw
high_for_fusion
FFT(low)
FFT(residual)
FFT(selected_residual)
```

重点看：

```text
selector 是否在目标区域更亮
selected_residual 是否比 residual 更干净
背景高频是否减少
```

---

## 13. Codex checklist

请 Codex 完成：

- [ ] 新增 `scripts/train_branch_probe.py`。
- [ ] 支持冻结 MIRFDNet，训练轻量 probe head。
- [ ] 支持分支：`low`, `residual`, `high_raw`, `low_residual`, `low_high_raw`, `low_residual_high_raw`。
- [ ] 输出 branch probe CSV。
- [ ] 新增 `mirfd/models/context_residual.py`。
- [ ] 实现 `ContextGuidedResidualSelector`。
- [ ] MIRFDBlock 支持 `selected_residual`。
- [ ] MIRFDBlock 支持 `use_context_residual_selector`。
- [ ] 支持 `selector_stages` 或 `use_context_residual_selector_by_stage`。
- [ ] loss 支持 `selector_supervision_weight`。
- [ ] 支持 selector GT dilation 后下采样。
- [ ] 诊断 CSV 增加 `selected_residual` 和 `selector` 相关字段。
- [ ] 可视化增加 selector / selected_residual。
- [ ] 新增 v2.5 配置：
  - `v2_5_probe_only`
  - `v2_5_cgrs_unsupervised`
  - `v2_5_cgrs_supervised`
  - `v2_5_cgrs_reference`
- [ ] 更新 README 和 `EXPERIMENT_RESULTS_AND_ANALYSIS.md`。

---

## 14. 第一轮最推荐执行顺序

### Step 1：Branch probe

先不要改主模型，先确认 low/residual/high_raw 的实际分割能力。

### Step 2：CGRS without reference

只使用：

```text
low + residual → selected_residual
```

验证低频 context 是否能指导 residual selection。

### Step 3：CGRS + weak selector supervision

如果无监督 selector 没提升，加轻量监督。

### Step 4：CGRS + stage-1 reference

最后再借鉴 AHFRM，加 shallow reference high-frequency。

---

## 15. 预期论文表述

如果 v2.5 有效，论文主线可写为：

```text
Mamba/SS2D is not treated as a pure low-pass filter. Instead, it provides a context representation that encodes global background structure. The residual between the input feature and the Mamba-induced context captures target-sensitive local discrepancies. A context-guided residual selector is then introduced to selectively preserve target-related discrepancies while suppressing clutter-like residual responses.
```

中文：

```text
本文并不将 Mamba/SS2D 简单视为纯低通滤波器，而是将其作为全局背景上下文表征。输入特征与 Mamba 诱导上下文之间的残差能够捕获目标敏感的局部差异。进一步地，本文提出上下文引导的残差选择模块，用于保留目标相关差异并抑制背景杂波残差。
```

这比“高频增强”更准确，也更符合当前所有实验结果。
