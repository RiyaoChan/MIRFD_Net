# MIRFD-Net：红外小目标分割专用的 Mamba-Residual Frequency Decoupling 网络设计说明

> 目标：将“输入特征与 Mamba 输出之间的残差信息可能包含被 Mamba 弱化的高频小目标细节”这一 idea 落地为可实现的红外小目标分割模型。  
> 适用数据集：NUDT-SIRST、IRSTD-1K、NUAA-SIRST。  
> 推荐任务形式：binary segmentation，输出单通道 mask。

---

## 1. 核心思想

红外小目标分割同时需要两类能力：

1. **全局背景建模能力**  
   红外图像背景通常包含天空、海面、云层、地面、建筑热纹理等大范围结构，需要模型理解全局上下文，抑制背景杂波。

2. **局部细节恢复能力**  
   红外小目标通常只有几个到几十个像素，表现为局部亮点、弱边缘、小区域突变，容易在全局建模过程中被平滑或弱化。

Mamba 具有长程建模能力，但在视觉特征中常表现出低频建模偏好。基于这一点，本模型不采用固定的 Laplace/Wavelet 显式分频，而是利用 Mamba 自身的低频偏好：

\[
F_m = \mathcal{M}(F)
\]

\[
F_l = \phi(F_m)
\]

\[
F_h = F - F_l
\]

其中：

- \(F\)：输入特征；
- \(\mathcal{M}(\cdot)\)：Mamba / SS2D / VMamba-style selective scan block；
- \(F_l\)：Mamba 输出对齐后的低频语义近似；
- \(F_h\)：输入特征与 Mamba 输出之间的残差，作为 high-frequency-enriched residual；
- \(\phi(\cdot)\)：1×1 Conv + Norm，用于通道和特征空间对齐。

核心假设：

> Mamba 输出 \(F_l\) 更偏全局、平滑、低频语义；残差 \(F_h = F - F_l\) 富集了被 Mamba 状态传播弱化的小目标边缘、局部亮点和细节突变。

---

## 2. 方法命名

推荐模型名：

**MIRFD-Net**  
**Mamba-Induced Residual Frequency Decoupling Network**

推荐核心模块名：

**MIRFD Block**  
**Mamba-Induced Residual Frequency Decoupling Block**

中文名称：

**Mamba 诱导的残差频率解耦网络**  
**Mamba 诱导的残差频率解耦模块**

---

## 3. 与已有频域 Mamba 方法的差异

已有相关方法通常是：

\[
F \xrightarrow{Laplace/Wavelet/Pooling/FFT} F_l, F_h
\]

然后：

\[
F_l \rightarrow Mamba
\]

\[
F_h \rightarrow Conv / Enhancement
\]

也就是说，它们先通过外部频域算子进行高低频分解，再把低频送入 Mamba。

MIRFD-Net 的设计是：

\[
F \xrightarrow{Mamba} F_l
\]

\[
F_h = F - F_l
\]

也就是说，本方法不依赖固定频率分解算子，而是利用 Mamba 自身的低频建模偏好生成自适应低频语义近似，再通过输入-输出残差恢复被弱化的高频细节。

---

## 4. 整体网络结构

建议采用轻量 U-Net / FPN 风格 encoder-decoder：

```text
Input Infrared Image
        |
      Stem
        |
   Encoder Stage 1  ---- skip_1 ----
        |                         |
   Encoder Stage 2  ---- skip_2 ----
        |                         |
   Encoder Stage 3  ---- skip_3 ----
        |                         |
   Encoder Stage 4  ---- skip_4 ----
        |                         |
     Decoder with High-frequency Residual Skip Fusion
        |
 Segmentation Head
        |
 Binary Mask
```

推荐阶段设置：

| Stage | Resolution | Module | Main Function |
|---|---:|---|---|
| Stem | 1/2 | Conv 3×3 | 初步局部特征提取 |
| Stage 1 | 1/2 or 1/4 | Conv Blocks | 保留浅层小目标位置和边缘 |
| Stage 2 | 1/4 or 1/8 | MIRFD Blocks | 初步全局-局部解耦 |
| Stage 3 | 1/8 or 1/16 | MIRFD Blocks | 背景上下文建模 + 高频残差恢复 |
| Stage 4 | 1/16 | MIRFD Blocks | 大范围背景抑制和语义增强 |
| Decoder | multi-scale | FPN/U-Net | 融合多尺度语义与小目标细节 |

注意：红外小目标非常小，不建议过早下采样到 1/32。默认最低尺度建议为 1/16。

---

## 5. MIRFD Block 设计

### 5.1 输入输出

输入：

\[
F \in \mathbb{R}^{B \times C \times H \times W}
\]

输出：

\[
F_{out} \in \mathbb{R}^{B \times C \times H \times W}
\]

同时可选返回：

- \(F_l\)：低频语义分支；
- \(F_h\)：高频残差分支；
- \(G\)：高频门控图；
- auxiliary prediction \(P_h\)。

---

### 5.2 Mamba 低频语义分支

```text
F_norm = Norm(F)
F_m    = Mamba2D(F_norm)
F_l    = Conv1x1(F_m)
```

公式：

\[
F_m = \mathcal{M}(\text{Norm}(F))
\]

\[
F_l = \phi(F_m)
\]

建议：

- Mamba2D 可使用 SS2D / VMamba-style block；
- 如果实现成本高，第一版可以用现成 VMamba block 或 selective_scan 代码；
- \(\phi\) 使用 `Conv2d(C, C, kernel_size=1) + BatchNorm2d / LayerNorm2d`；
- 输出尺度必须与输入 \(F\) 一致，才能做残差相减。

---

### 5.3 Mamba-induced 高频残差分支

```text
R   = F - F_l
F_h = HFE(R)
```

公式：

\[
R = F - F_l
\]

\[
F_h = \mathcal{H}(R)
\]

其中 \(\mathcal{H}\) 是 high-frequency residual enhancement module，建议包含多尺度 depth-wise convolution：

```text
HFE(R):
    r1 = DWConv3x3(R)
    r2 = DWConv5x5(R)
    r3 = PWConv1x1(concat(r1, r2))
    return r3
```

建议实现：

```python
class HighFrequencyEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.pw = nn.Conv2d(dim * 2, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x3 = self.dw3(x)
        x5 = self.dw5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x
```

---

### 5.4 目标感知高频门控

高频残差中不仅有小目标，也有背景杂波、云边缘、建筑边缘和噪声。因此需要 gate 过滤。

```text
G = sigmoid(Conv1x1(concat(F_l, R)))
F_h_hat = G * F_h
```

公式：

\[
G = \sigma(\psi([F_l, R]))
\]

\[
\hat{F}_h = G \odot F_h
\]

其中：

- \(G\)：目标感知高频门控；
- \(\psi\)：1×1 Conv；
- \(\odot\)：逐元素乘法。

建议：

```python
class TargetAwareGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, low, residual):
        g = self.gate(torch.cat([low, residual], dim=1))
        return g
```

---

### 5.5 低频-高频融合

推荐两种融合方式。

#### 方案 A：Concat Fusion

```text
F_out = Conv1x1(concat(F_l, F_h_hat)) + F
```

公式：

\[
F_{out} = \eta([F_l, \hat{F}_h]) + F
\]

优点：稳定、容易实现。

#### 方案 B：Residual Compensation Fusion

```text
F_out = F_l + gamma * F_h_hat + F
```

公式：

\[
F_{out} = F + F_l + \gamma \hat{F}_h
\]

其中 \(\gamma\) 是可学习标量，建议初始化为 0 或 0.1。

第一版推荐使用方案 A。

---

## 6. MIRFD Block 伪代码

```python
class MIRFDBlock(nn.Module):
    def __init__(self, dim, mamba_block):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.mamba = mamba_block(dim)

        self.align = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

        self.hfe = HighFrequencyEnhancer(dim)
        self.gate = TargetAwareGate(dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x, return_branches=False):
        # Low-frequency semantic approximation induced by Mamba
        fm = self.mamba(self.norm(x))
        fl = self.align(fm)

        # Mamba-induced high-frequency enriched residual
        r = x - fl
        fh = self.hfe(r)

        # Target-aware high-frequency filtering
        g = self.gate(fl, r)
        fh_hat = g * fh

        # Fuse low-frequency semantics and high-frequency residual details
        out = self.fuse(torch.cat([fl, fh_hat], dim=1)) + x

        if return_branches:
            return out, fl, fh_hat, r, g
        return out
```

---

## 7. Decoder 设计

建议使用 FPN + U-Net skip fusion。

每个尺度的 encoder 输出包括：

```text
E_s: MIRFD output
H_s: high-frequency residual branch output
```

Decoder 融合方式：

\[
D_s = Fuse(Up(D_{s+1}), E_s, H_s)
\]

伪代码：

```python
class DecoderBlock(nn.Module):
    def __init__(self, dim_high, dim_skip, dim_out):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim_high + dim_skip * 2, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )

    def forward(self, x_high, skip, high_residual):
        x_high = F.interpolate(x_high, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_high, skip, high_residual], dim=1)
        return self.fuse(x)
```

最终 segmentation head：

```python
self.head = nn.Sequential(
    nn.Conv2d(dec_dim, dec_dim // 2, 3, padding=1),
    nn.BatchNorm2d(dec_dim // 2),
    nn.GELU(),
    nn.Conv2d(dec_dim // 2, 1, 1)
)
```

输出 logits，不在模型内部 sigmoid；训练时使用 `BCEWithLogitsLoss`。

---

## 8. Loss 设计

### 8.1 基础分割损失

推荐：

\[
\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}
\]

Dice Loss：

\[
\mathcal{L}_{Dice} = 1 - \frac{2\sum p y + \epsilon}{\sum p + \sum y + \epsilon}
\]

其中：

- \(p = sigmoid(logits)\)
- \(y\)：GT mask。

---

### 8.2 高频分支辅助监督

对最后 1-3 个尺度的高频残差分支加 auxiliary segmentation head：

\[
P_h^s = Head_h(F_h^s)
\]

\[
\mathcal{L}_{aux} = \sum_s \left(\mathcal{L}_{BCE}(P_h^s, Y_s) + \mathcal{L}_{Dice}(P_h^s, Y_s)\right)
\]

其中 \(Y_s\) 是下采样到对应尺度的 GT mask。

目的：

> 约束高频残差分支不要泛泛增强所有高频，而是重点增强与小目标 mask 对应的目标响应。

---

### 8.3 频谱软约束

FFT 不参与 forward 特征生成，只作为训练正则项。

对 Mamba 低频分支：

\[
\mathcal{L}_{low}
=
\frac{\|FFT(F_l)\cdot M_{high}\|_1}
{\|FFT(F_l)\|_1 + \epsilon}
\]

含义：惩罚低频分支中不应过多存在的高频能量。

对残差高频分支：

\[
\mathcal{L}_{high}
=
\frac{\|FFT(F_h)\cdot M_{low}\|_1}
{\|FFT(F_h)\|_1 + \epsilon}
\]

含义：惩罚高频残差分支中不应过多存在的低频能量。

总损失：

\[
\mathcal{L}
=
\mathcal{L}_{seg}
+
\lambda_{low}\mathcal{L}_{low}
+
\lambda_{high}\mathcal{L}_{high}
+
\lambda_{aux}\mathcal{L}_{aux}
\]

推荐初始权重：

```yaml
lambda_low: 0.01
lambda_high: 0.01
lambda_aux: 0.2
```

注意：

- 频谱约束是软约束，不要强制完全分离高低频；
- \(\lambda_{low}\) 和 \(\lambda_{high}\) 不宜过大；
- 可以先不加频谱约束训练 baseline，再加该约束做消融。

---

## 9. FFT 频谱约束实现建议

输入特征：

```text
feat: [B, C, H, W]
```

建议对空间维度做 FFT：

```python
fft = torch.fft.fft2(feat, dim=(-2, -1), norm="ortho")
fft = torch.fft.fftshift(fft, dim=(-2, -1))
mag = torch.abs(fft)
```

生成频率 mask：

```python
def build_frequency_masks(h, w, low_radius_ratio=0.25, device="cuda"):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    cy, cx = h // 2, w // 2
    dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    radius = low_radius_ratio * min(h, w)
    m_low = (dist <= radius).float()
    m_high = 1.0 - m_low
    return m_low[None, None, :, :], m_high[None, None, :, :]
```

Loss 计算：

```python
def spectral_regularization(low_feat, high_feat, low_radius_ratio=0.25, eps=1e-6):
    _, _, h, w = low_feat.shape
    m_low, m_high = build_frequency_masks(h, w, low_radius_ratio, low_feat.device)

    low_fft = torch.fft.fftshift(
        torch.fft.fft2(low_feat, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1)
    )
    high_fft = torch.fft.fftshift(
        torch.fft.fft2(high_feat, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1)
    )

    low_mag = torch.abs(low_fft)
    high_mag = torch.abs(high_fft)

    loss_low = (low_mag * m_high).sum() / (low_mag.sum() + eps)
    loss_high = (high_mag * m_low).sum() / (high_mag.sum() + eps)

    return loss_low, loss_high
```

---

## 10. 数据集与训练建议

### 10.1 数据集

建议支持：

- NUDT-SIRST
- IRSTD-1K
- NUAA-SIRST

统一格式：

```text
dataset_root/
    images/
        xxx.png
    masks/
        xxx.png
    train.txt
    val.txt
    test.txt
```

mask 要求：

- binary mask；
- 前景为 1，背景为 0；
- 训练前统一归一化到 `{0, 1}`。

---

### 10.2 输入尺寸

推荐：

```yaml
input_size: 256
```

如果 GPU 允许，可做：

```yaml
input_size: 384
```

红外小目标较小，避免过强 resize 导致目标消失。

---

### 10.3 数据增强

推荐增强：

```yaml
augmentations:
  - random_horizontal_flip
  - random_vertical_flip
  - random_rotate_90
  - random_crop_or_resize
  - brightness_contrast_jitter
  - gaussian_noise_light
```

谨慎使用：

```yaml
strong_blur: false
large_scale_downsample: false
heavy_cutmix: false
```

原因：目标太小，强模糊或过强缩放容易破坏小目标。

---

### 10.4 训练配置

推荐初始配置：

```yaml
optimizer: AdamW
base_lr: 0.0003
weight_decay: 0.01
batch_size: 8
epochs: 300
scheduler: cosine
warmup_epochs: 5
loss:
  bce: 1.0
  dice: 1.0
  lambda_low: 0.01
  lambda_high: 0.01
  lambda_aux: 0.2
```

若使用 3090：

```yaml
input_size: 256
batch_size: 8-16
amp: true
```

若使用 5090：

```yaml
input_size: 384
batch_size: 8-16
amp: true
```

---

## 11. Evaluation 指标

建议至少报告：

### Segmentation metrics

```text
IoU
nIoU
Dice / F1
Precision
Recall
```

### Detection-style metrics for small target

```text
Pd: probability of detection
Fa: false alarm rate
```

可根据 connected components 从预测 mask 中提取目标点/区域，计算目标级检出率与虚警率。

---

## 12. 关键消融实验

必须做以下消融，否则 idea 不容易站住。

| Variant | Purpose |
|---|---|
| Baseline U-Net/FPN | 基础对照 |
| Baseline + Mamba | 验证单纯加 Mamba 是否有效 |
| Baseline + MIRFD | 验证输入-Mamba输出残差是否有效 |
| MIRFD without Gate | 验证目标感知高频门控是否有效 |
| MIRFD without Spectral Loss | 验证频谱软约束是否有效 |
| MIRFD without Aux Head | 验证高频分支辅助监督是否有效 |
| Replace MIRFD residual with AvgPool residual | 对比普通 pooling 高频残差 |
| Replace MIRFD residual with Laplace high-frequency | 对比显式 Laplace 高频 |
| Replace MIRFD residual with Sobel edge | 对比边缘先验 |
| Replace MIRFD residual with Wavelet high-frequency | 对比小波高频 |
| MIRFD in Stage 2 only / Stage 3 only / Stage 2-4 | 找最佳插入位置 |

最关键对比：

\[
F_h = F - Mamba(F)
\]

vs.

\[
F_h = F - Up(AvgPool(F))
\]

vs.

\[
F_h = LaplaceHigh(F)
\]

vs.

\[
F_h = Sobel(F)
\]

vs.

\[
F_h = WaveletHigh(F)
\]

如果 MIRFD residual 更好，说明该方法不是普通高频增强，而是 Mamba-induced residual 有任务价值。

---

## 13. 可视化实验

建议做四类可视化。

### 13.1 频谱可视化

展示：

```text
Input feature F
Mamba output F_l
Residual F_h = F - F_l
```

对应 Fourier spectrum：

- \(F\)：频率分布较混合；
- \(F_l\)：中心低频能量更集中；
- \(F_h\)：外围高频能量相对更强。

### 13.2 高频能量比例

计算：

\[
R_{high} = \frac{\|FFT(F)\cdot M_{high}\|_1}{\|FFT(F)\|_1}
\]

比较：

```text
F
F_l
F_h
```

预期：

```text
High-frequency ratio: F_h > F > F_l
```

### 13.3 特征响应图

展示：

```text
Mamba low-frequency branch response
High-frequency residual branch response
Gate map
Final prediction
GT mask
```

目标：

- 证明 \(F_h\) 更关注小目标局部；
- 证明 gate 能减少背景杂波；
- 证明融合后减少漏检。

### 13.4 失败案例分析

建议展示：

```text
Baseline Mamba 漏检
MIRFD 恢复小目标响应
```

以及：

```text
普通高频增强误检背景边缘
MIRFD gate 抑制误检
```

---

## 14. 推荐代码结构

```text
project/
    configs/
        mirfd_nudt.yaml
        mirfd_irstd1k.yaml
        mirfd_nuaa.yaml

    datasets/
        sirst_dataset.py
        transforms.py

    models/
        mirfd_net.py
        mirfd_block.py
        mamba2d.py
        decoder.py
        losses.py

    utils/
        metrics.py
        visualize_fft.py
        visualize_features.py

    train.py
    test.py
    infer.py
```

---

## 15. MVP 实现顺序

建议 Codex 按以下顺序实现，避免一次性写太复杂。

### Step 1：实现 baseline segmentation framework

- Conv Stem
- 4-stage encoder
- FPN/U-Net decoder
- BCE + Dice loss
- IoU/nIoU metric

### Step 2：接入 Mamba block

- 实现或调用 Mamba2D block
- 先做 Baseline + Mamba

### Step 3：实现 MIRFD Block

- Mamba low-frequency branch
- residual branch \(R = F - F_l\)
- HighFrequencyEnhancer
- Low-high fusion

### Step 4：加入 TargetAwareGate

- 生成 gate map
- 抑制高频背景杂波

### Step 5：加入 auxiliary high-frequency supervision

- 高频分支 auxiliary head
- 多尺度 mask 下采样

### Step 6：加入 spectral regularization

- FFT loss
- low/high mask
- loss 权重配置

### Step 7：实现 ablation switches

配置文件中支持：

```yaml
model:
  use_mamba: true
  use_mirfd: true
  use_hf_gate: true
  use_aux_head: true
  use_spectral_loss: true
  residual_type: mamba_residual  # options: mamba_residual, avgpool, laplace, sobel, wavelet
```

---

## 16. 第一版推荐模型配置

```yaml
model:
  name: MIRFDNet
  in_channels: 1
  num_classes: 1
  input_size: 256

  encoder:
    dims: [32, 64, 128, 256]
    depths: [2, 2, 2, 2]
    use_mirfd_stages: [false, true, true, true]
    min_stride: 16

  mirfd:
    use_gate: true
    use_learnable_gamma: false
    fusion: concat
    hfe_kernels: [3, 5]

  decoder:
    type: unet_fpn
    dim: 64
    use_high_residual_skip: true

loss:
  bce_weight: 1.0
  dice_weight: 1.0
  aux_weight: 0.2
  spectral_low_weight: 0.01
  spectral_high_weight: 0.01
  spectral_low_radius_ratio: 0.25

train:
  optimizer: AdamW
  lr: 0.0003
  weight_decay: 0.01
  epochs: 300
  batch_size: 8
  amp: true
  scheduler: cosine
  warmup_epochs: 5
```

---

## 17. 论文贡献点建议表述

### Contribution 1

提出 Mamba-induced residual frequency decoupling，用 Mamba 输出作为自适应低频语义近似，并通过输入-Mamba输出残差恢复被状态空间传播弱化的高频小目标细节。

### Contribution 2

设计 target-aware high-frequency residual enhancement，通过低频背景上下文筛选高频残差，增强小目标响应并抑制背景杂波误增强。

### Contribution 3

提出 soft spectral regularization，使 Mamba 诱导分支和残差分支分别保持低频和高频偏好，将频率解耦从观察启发转化为可训练、可验证机制。

### Contribution 4

在 NUDT-SIRST、IRSTD-1K、NUAA-SIRST 上验证模型，并通过跨数据集泛化、频谱可视化和消融实验证明该机制对红外小目标分割有效。

---

## 18. 需要避免的表述

不要写：

> Mamba 是严格的高低频分频器。

推荐写：

> Mamba exhibits a low-frequency modeling preference in visual feature propagation.

不要写：

> \(F - Mamba(F)\) 就是纯高频。

推荐写：

> The input-output residual is treated as a high-frequency-enriched residual that contains details attenuated during Mamba-based state propagation.

不要写：

> 频谱约束强制完全高低频分离。

推荐写：

> The spectral constraint softly encourages the Mamba-induced branch and the residual branch to exhibit low-frequency and high-frequency preferences, respectively.

---

## 19. 最终实现目标

Codex 最终需要实现：

1. `MIRFDNet`
2. `MIRFDBlock`
3. `HighFrequencyEnhancer`
4. `TargetAwareGate`
5. `spectral_regularization`
6. `BCE + Dice + Aux + Spectral loss`
7. dataset loaders for NUDT-SIRST / IRSTD-1K / NUAA-SIRST
8. training / testing / inference scripts
9. ablation switches
10. FFT visualization tools

---

## 20. 最小可运行版本

如果先做最小可运行版本，优先实现：

```text
MIRFD-Net = Conv Stem + Encoder with MIRFD Block + U-Net Decoder
Loss = BCE + Dice
Metrics = IoU + nIoU
```

然后逐步加入：

```text
+ TargetAwareGate
+ Aux Head
+ Spectral Loss
+ Ablation residual alternatives
+ FFT visualization
```

---

## 21. 一句话总结

MIRFD-Net 的核心不是使用外部分频算子预先拆分高低频，而是利用 Mamba 的低频建模偏好，将 Mamba 输出作为自适应低频语义近似，并通过输入-Mamba输出残差恢复被弱化的红外小目标高频细节，从而实现全局背景抑制与局部小目标增强的协同建模。
