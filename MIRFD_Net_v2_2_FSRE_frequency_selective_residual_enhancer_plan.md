# MIRFD-Net v2.2 改进计划：Frequency-Selective Residual Enhancer（FSRE）

> 目的：把当前基于卷积的 HFE（High-Frequency Enhancer）扩展为局部频域选择性残差增强模块，使 high branch 从“空间卷积高频增强”转向“目标相关频带选择”。  
> 背景：当前统计显示 stage-1 的 `residual = F - low` 已经高度关注目标区域，说明最初的 Mamba-induced residual 假设成立；但经过卷积 HFE 后，stage-1 `high_raw_fg_bg` 反而低于 `residual_fg_bg`，说明普通卷积可能削弱了原始 residual 的目标选择性。

---

## 1. 核心判断

当前问题不是 residual 无效，而是后续 high feature 学习不够有针对性。

当前流程：

```text
low0 = Align(SS2D(Norm(F)))
low = LowSmooth(low0)
residual = F - low
high_raw = HFE(residual)
high_hat = Gate(high_raw)
```

已有诊断说明：

```text
stage-1 residual 非常目标相关；
stage-1 high_raw 反而低于 residual；
stage-2 high_raw 有一定价值；
gate 没有稳定提升 high_hat 的 fg/bg；
stage-3/stage-4 不适合作为 decoder high skip。
```

因此下一步建议：

```text
stage-1: 尽量保留原始 residual，不要过度处理；
stage-2: 用局部频域频带选择替代普通卷积 HFE；
decoder: 优先使用 high_raw，而不是 high_hat。
```

---

## 2. 是否是在局部频域里做频带选择？

是的，建议优先在 **局部频域** 中做频带选择，而不是全局 FFT。

### 为什么不是全局 FFT？

全局 FFT 会把整张图的频率混在一起。小目标、地平线、云层边缘、海面纹理、建筑边缘都会进入同一个全局频谱。  
如果直接增强全局高频，可能同时增强目标和背景杂波。

### 为什么是局部频域？

红外小目标是局部点状或小斑块结构。局部频域窗口更适合判断：

```text
这个局部窗口里是否存在目标相关的点状中高频突变？
```

推荐流程：

```text
residual
-> 划分局部窗口，例如 8×8 或 16×16
-> 每个窗口做 FFT
-> 按 radial frequency bands 统计频带能量
-> 学习每个频带的权重
-> 频域加权
-> inverse FFT 回到空间域
-> 得到 frequency-enhanced residual
```

---

## 3. 频带选择是什么意思？

不要学习完整的 `H×W` 频域 mask。建议把局部窗口频域划分成几个 radial bands：

```text
Band 1: low frequency
Band 2: mid-low frequency
Band 3: mid-high frequency
Band 4: high frequency
```

构建固定频带 mask：

```text
M1, M2, M3, M4
```

网络学习对应权重：

```text
w1, w2, w3, w4
```

频域调制：

```text
A(f) = w1*M1 + w2*M2 + w3*M3 + w4*M4
FFT_filtered = FFT(residual_window) * (1 + A(f))
```

注意：红外小目标可能最依赖 **中高频**，不一定是最高频。最高频也可能是噪声或背景纹理。因此 FSRE 的目标不是增强所有高频，而是学习目标相关频带。

---

## 4. 可学习缩放系数是否可学习？

是的，建议使用可学习缩放系数 `gamma`。

FSRE 输出采用 residual-style：

```text
R_freq = frequency_branch(R)
high_raw = R + gamma * R_freq
```

其中 `gamma` 是可学习参数：

```python
self.gamma = nn.Parameter(torch.tensor(gamma_init))
```

推荐初始值：

```yaml
fsre_gamma_init: 0.1
```

forward 中：

```python
gamma = torch.clamp(self.gamma, 0.0, 1.0)
out = residual + gamma * freq_enhanced
```

### 为什么 gamma 要可学习？

因为频域增强一开始不能太强。  
如果一开始就大幅增强频域分量，很容易放大背景纹理或噪声。

`gamma_init=0.1` 的含义是：

```text
先主要保留原始 residual；
频域增强作为轻量补充；
训练中模型自己学习是否需要增强、增强多少。
```

第一版建议 `gamma` 用标量参数。后续可以尝试逐通道参数：

```python
self.gamma = nn.Parameter(torch.ones(1, C, 1, 1) * gamma_init)
```

---

## 5. 新模块：FSRE

建议新增文件：

```text
mirfd/models/frequency_enhancer.py
```

新增模块：

```python
class FrequencySelectiveResidualEnhancer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        window_size: int = 8,
        gamma_init: float = 0.1,
        norm: str = "bn",
    ):
        ...
```

建议配置：

```yaml
mirfd:
  high_enhancer_type: freq_window   # identity | conv_hfe | freq_window
  fsre_num_bands: 4
  fsre_window_size: 8
  fsre_gamma_init: 0.1

model:
  stage1_high_enhancer_type: identity  # identity | conv_hfe | freq_window
  decoder_high_source: high_raw        # high_raw | high_hat | residual
```

---

## 6. FSRE 伪代码

```python
class FrequencySelectiveResidualEnhancer(nn.Module):
    def __init__(self, dim, num_bands=4, window_size=8, gamma_init=0.1, norm="bn"):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.window_size = window_size

        self.band_mlp = nn.Sequential(
            nn.Linear(num_bands, num_bands),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands, num_bands),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: residual, [B, C, H, W]
        B, C, H, W = x.shape
        ws = self.window_size

        # pad to window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x_pad.shape[-2:]

        # window partition: [B*nW, C, ws, ws]
        x_win = x_pad.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x_win = x_win.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_win = x_win.view(-1, C, ws, ws)

        # FFT
        fft = torch.fft.fft2(x_win, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft)

        # radial band masks: [num_bands, ws, ws]
        masks = build_radial_band_masks(ws, ws, self.num_bands, x.device)
        masks = masks.to(dtype=mag.dtype)

        # band energy: [B*nW, C, num_bands]
        energy_list = []
        for i in range(self.num_bands):
            m = masks[i].view(1, 1, ws, ws)
            e = (mag * m).sum(dim=(-2, -1)) / (m.sum() + 1e-6)
            energy_list.append(e)
        band_energy = torch.stack(energy_list, dim=-1)

        # descriptor: [B*nW, num_bands]
        desc = band_energy.mean(dim=1)

        # weights: [B*nW, num_bands]
        weights = self.band_mlp(desc)

        # frequency weight map: [B*nW, 1, ws, ws]
        freq_weight = 0
        for i in range(self.num_bands):
            freq_weight = freq_weight + weights[:, i].view(-1, 1, 1, 1) * masks[i].view(1, 1, ws, ws)

        # frequency modulation
        fft_filtered = fft * (1.0 + freq_weight)
        fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        x_freq = torch.fft.ifft2(fft_filtered, dim=(-2, -1)).real

        # reverse windows
        x_freq = x_freq.view(B, Hp // ws, Wp // ws, C, ws, ws)
        x_freq = x_freq.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_freq = x_freq.view(B, C, Hp, Wp)
        x_freq = x_freq[:, :, :H, :W]

        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        out = x + gamma * self.proj(x_freq)
        return out
```

---

## 7. radial band mask 构建

```python
def build_radial_band_masks(H, W, num_bands, device):
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy, cx = H // 2, W // 2
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist = dist / (dist.max() + 1e-6)

    masks = []
    edges = torch.linspace(0, 1, num_bands + 1, device=device)
    for i in range(num_bands):
        if i == num_bands - 1:
            mask = (dist >= edges[i]) & (dist <= edges[i + 1])
        else:
            mask = (dist >= edges[i]) & (dist < edges[i + 1])
        masks.append(mask.float())

    return torch.stack(masks, dim=0)
```

建议后续做缓存，避免每次 forward 重建 mask。第一版可以先不缓存，优先保证正确性。

---

## 8. 如何接入现有模型

建议新增 high enhancer factory：

```python
def build_high_enhancer(
    enhancer_type,
    dim,
    norm="bn",
    fsre_num_bands=4,
    fsre_window_size=8,
    fsre_gamma_init=0.1,
):
    if enhancer_type == "identity":
        return nn.Identity()
    if enhancer_type == "conv_hfe":
        return HighFrequencyEnhancer(dim, norm=norm)
    if enhancer_type == "freq_window":
        return FrequencySelectiveResidualEnhancer(
            dim=dim,
            num_bands=fsre_num_bands,
            window_size=fsre_window_size,
            gamma_init=fsre_gamma_init,
            norm=norm,
        )
    raise ValueError(f"Unsupported high_enhancer_type: {enhancer_type}")
```

MIRFD Block 中：

```python
self.high_enhancer = build_high_enhancer(...)
```

forward 中：

```python
high_raw = self.high_enhancer(residual)
```

如果保留 `concat_proj`：

```python
enhanced = self.high_enhancer(residual)
high_raw = self.high_proj(torch.cat([residual, enhanced], dim=1))
```

但第一版建议直接：

```python
high_raw = self.high_enhancer(residual)
```

因为 FSRE 内部已经是 residual-style：

```text
out = residual + gamma * freq_enhanced
```

---

## 9. Stage-1 high skip 接入建议

统计显示 stage-1 residual 已经非常目标相关，因此第一版不建议对 stage-1 过度处理。

新增：

```yaml
model:
  stage1_high_enhancer_type: identity | conv_hfe | freq_window
```

第一轮推荐：

```yaml
stage1_high_enhancer_type: identity
```

即 stage-1 直接使用：

```text
stage1_high = stage1_residual
```

第二轮再试：

```yaml
stage1_high_enhancer_type: freq_window
```

---

## 10. 推荐实验配置

### Experiment A：Identity residual baseline

```yaml
model:
  high_skip_stages: [1, 2]
  decoder_high_source: high_raw
  stage1_high_enhancer_type: identity

mirfd:
  high_enhancer_type: identity

loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：验证直接用 residual 是否已经足够。

---

### Experiment B：Conv-HFE baseline

```yaml
model:
  high_skip_stages: [1, 2]
  decoder_high_source: high_raw
  stage1_high_enhancer_type: conv_hfe

mirfd:
  high_enhancer_type: conv_hfe

loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：当前卷积 HFE 对照。

---

### Experiment C：Window FSRE

```yaml
model:
  high_skip_stages: [1, 2]
  decoder_high_source: high_raw
  stage1_high_enhancer_type: freq_window

mirfd:
  high_enhancer_type: freq_window
  fsre_num_bands: 4
  fsre_window_size: 8
  fsre_gamma_init: 0.1

loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：验证局部频域频带选择是否优于卷积 HFE。

---

### Experiment D：Stage-1 identity + Stage-2 FSRE

```yaml
model:
  high_skip_stages: [1, 2]
  decoder_high_source: high_raw
  stage1_high_enhancer_type: identity

mirfd:
  high_enhancer_type: freq_window
  fsre_num_bands: 4
  fsre_window_size: 8
  fsre_gamma_init: 0.1

loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```

目的：保留最干净的 stage-1 residual，只在 stage-2 使用 FSRE。  
当前最推荐优先跑这个。

---

## 11. 诊断指标

FSRE 加入后，继续使用 `diagnose_feature_statistics.py`。重点看：

```text
high_raw_fg_bg 是否高于 residual_fg_bg
false_alarm_rate 是否下降
stage-2 high_raw_fg_bg 是否提升
stage-1 high_raw_fg_bg 是否不下降
R_high_high_raw 是否合理提升
```

如果 FSRE 有效，应看到：

```text
stage-2 high_raw_fg_bg >= stage-2 residual_fg_bg
stage-1 high_raw_fg_bg 不明显低于 residual_fg_bg
pred_has_false_alarm 降低
final IoU / nIoU / Pd 提升
```

---

## 12. 论文表述建议

不要写：

```text
We enhance all high-frequency components.
```

推荐写：

```text
We introduce a frequency-selective residual enhancer that learns adaptive band-wise modulation in local frequency windows, enabling target-relevant mid-high frequency cues to be emphasized while avoiding indiscriminate amplification of clutter-like high-frequency residuals.
```

中文：

```text
我们提出频率选择性残差增强模块，在局部频域窗口中学习自适应频带调制，从而突出目标相关的中高频线索，并避免无差别放大背景杂波高频残差。
```

也不要把 residual 写成纯高频：

```text
Mamba-induced residual is high-frequency-enriched and target-sensitive at shallow stages, but it is not a pure frequency component.
```

---

## 13. Codex 修改 checklist

请 Codex 完成：

- [ ] 新增 `mirfd/models/frequency_enhancer.py`。
- [ ] 实现 `FrequencySelectiveResidualEnhancer`。
- [ ] 实现 `build_radial_band_masks`。
- [ ] 支持 window FFT，默认 `window_size=8`。
- [ ] 支持 `fsre_num_bands`，默认 4。
- [ ] 支持可学习 `gamma`，默认 `gamma_init=0.1`。
- [ ] 新增 `high_enhancer_type`: `identity | conv_hfe | freq_window`。
- [ ] 新增 `stage1_high_enhancer_type`: `identity | conv_hfe | freq_window`。
- [ ] 新增 `decoder_high_source`: `high_raw | high_hat | residual`。
- [ ] MIRFD Block 支持使用 FSRE 替换 HFE。
- [ ] Stage-1 high skip 支持 identity residual 或 FSRE。
- [ ] 新增实验配置：
  - `v2_2_identity_residual`
  - `v2_2_conv_hfe_high_raw`
  - `v2_2_window_fsre`
  - `v2_2_stage1_identity_stage2_fsre`
- [ ] 更新 README 的 MIRFD Block 说明。
- [ ] 更新 EXPERIMENT_RESULTS_AND_ANALYSIS.md，说明 v2.2 的目标是 frequency-selective residual enhancement。

---

## 14. 最终推荐

第一轮最推荐：

```text
Experiment D: Stage-1 identity + Stage-2 FSRE
```

原因：

1. stage-1 residual 已被统计证明非常目标相关，不应过度处理；
2. stage-2 residual/high_raw 仍有提升空间；
3. 局部频域 FSRE 更适合 stage-2 做目标相关频带选择；
4. decoder 直接使用 high_raw，避免 gate 干扰。

推荐核心配置：

```yaml
model:
  high_skip_stages: [1, 2]
  decoder_high_source: high_raw
  stage1_high_enhancer_type: identity

mirfd:
  high_enhancer_type: freq_window
  fsre_num_bands: 4
  fsre_window_size: 8
  fsre_gamma_init: 0.1

loss:
  spectral_low_weight: 0.0
  spectral_high_weight: 0.0
```
