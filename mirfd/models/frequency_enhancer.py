from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvNormAct


def build_radial_band_masks(
    height: int,
    width: int,
    num_bands: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy, cx = height // 2, width // 2
    dist = torch.sqrt((yy - cy).pow(2) + (xx - cx).pow(2))
    dist = dist / (dist.max() + 1e-6)

    masks = []
    edges = torch.linspace(0.0, 1.0, num_bands + 1, device=device)
    for index in range(num_bands):
        if index == num_bands - 1:
            mask = (dist >= edges[index]) & (dist <= edges[index + 1])
        else:
            mask = (dist >= edges[index]) & (dist < edges[index + 1])
        masks.append(mask.float())
    return torch.stack(masks, dim=0)


class FrequencySelectiveResidualEnhancer(nn.Module):
    """Local-window frequency-band residual enhancer for MIRFD high branches."""

    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        window_size: int = 8,
        gamma_init: float = 0.1,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.dim = dim
        self.num_bands = int(num_bands)
        self.window_size = int(window_size)
        self.band_mlp = nn.Sequential(
            nn.Linear(self.num_bands, self.num_bands),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_bands, self.num_bands),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.proj = ConvNormAct(dim, dim, kernel_size=1, padding=0, norm=norm)
        self.register_buffer("_mask_cache", torch.empty(0), persistent=False)

    def _band_masks(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._mask_cache.numel() == 0 or self._mask_cache.device != device:
            self._mask_cache = build_radial_band_masks(
                self.window_size,
                self.window_size,
                self.num_bands,
                device=device,
            )
        return self._mask_cache.to(dtype=dtype)

    @staticmethod
    def _pad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        _, _, height, width = x.shape
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        x_pad = self._pad(x, pad_h, pad_w)
        padded_h, padded_w = x_pad.shape[-2:]

        x_win = x_pad.view(batch, channels, padded_h // ws, ws, padded_w // ws, ws)
        x_win = x_win.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_win = x_win.view(-1, channels, ws, ws)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_win_float = x_win.float()
            fft = torch.fft.fft2(x_win_float, dim=(-2, -1))
            fft = torch.fft.fftshift(fft, dim=(-2, -1))
            mag = torch.abs(fft)
            masks = self._band_masks(x.device, mag.dtype)
            masks_view = masks.view(1, 1, self.num_bands, ws, ws)
            band_energy = (mag.unsqueeze(2) * masks_view).sum(dim=(-2, -1))
            band_energy = band_energy / (masks_view.sum(dim=(-2, -1)) + 1e-6)

            desc = band_energy.mean(dim=1)
            weights = self.band_mlp(desc)
            freq_weight = torch.einsum("nb,bhw->nhw", weights, masks).unsqueeze(1)

            fft_filtered = fft * (1.0 + freq_weight)
            fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
            x_freq = torch.fft.ifft2(fft_filtered, dim=(-2, -1)).real

            x_freq = x_freq.view(batch, padded_h // ws, padded_w // ws, channels, ws, ws)
            x_freq = x_freq.permute(0, 3, 1, 4, 2, 5).contiguous()
            x_freq = x_freq.view(batch, channels, padded_h, padded_w)
            x_freq = x_freq[:, :, :height, :width]

        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        return x + gamma * self.proj(x_freq.to(dtype=x.dtype))


class FFCStyleHighFreqGate(nn.Module):
    """High-frequency spectral gate adapted from the SCTransNet FFC Fourier unit."""

    def __init__(self, dim: int, reduction: int = 4, highfreq_threshold: float = 0.5) -> None:
        super().__init__()
        hidden = max(dim // max(int(reduction), 1), 1)
        self.highfreq_threshold = float(highfreq_threshold)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.register_buffer("_mask_cache", torch.empty(0), persistent=False)
        self._mask_shape: tuple[int, int] | None = None

    def _highfreq_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._mask_cache.numel() == 0 or self._mask_cache.device != device or self._mask_shape != (height, width):
            y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
            x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            radius = torch.sqrt(xx.pow(2) + yy.pow(2)).clamp(0.0, 1.0)
            self._mask_cache = (radius >= self.highfreq_threshold).float()
            self._mask_shape = (height, width)
        return self._mask_cache.to(dtype=dtype)

    def forward(self, fft_feat: torch.Tensor) -> torch.Tensor:
        amp = torch.abs(fft_feat)
        height, width = amp.shape[-2:]
        mask = self._highfreq_mask(height, width, amp.device, amp.real.dtype).view(1, 1, height, width)
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        high_mean = (amp * mask).sum(dim=(-2, -1), keepdim=True) / denom
        return 1.0 + torch.tanh(self.fc2(self.act(self.fc1(high_mean))))


class FFCFrequencyResidualEnhancer(nn.Module):
    """FFC-style global Fourier enhancer for MIRFD residual high branches.

    This follows the SCTransNet FFC pattern: rFFT, real/imaginary channel mixing
    by 1x1 convolution, optional high-frequency spectral gate, inverse rFFT, and
    spatial fusion with a local branch.
    """

    def __init__(
        self,
        dim: int,
        gamma_init: float = 0.1,
        norm: str = "batch",
        fft_norm: str = "ortho",
        use_highfreq_gate: bool = True,
        highfreq_threshold: float = 0.5,
        gate_reduction: int = 4,
        local_kernel: int = 3,
    ) -> None:
        super().__init__()
        if local_kernel <= 0 or local_kernel % 2 == 0:
            raise ValueError("local_kernel must be a positive odd integer")
        self.dim = dim
        self.fft_norm = fft_norm
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
        )
        self.highfreq_gate = (
            FFCStyleHighFreqGate(dim, reduction=gate_reduction, highfreq_threshold=highfreq_threshold)
            if use_highfreq_gate
            else None
        )
        self.local = ConvNormAct(
            dim,
            dim,
            kernel_size=local_kernel,
            padding=local_kernel // 2,
            norm=norm,
            groups=dim,
        )
        self.fuse = ConvNormAct(dim * 2, dim, kernel_size=1, padding=0, norm=norm)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _fourier_branch(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            ffted = torch.fft.rfftn(x_float, dim=(-2, -1), norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=2)
            ffted = ffted.view(batch, channels * 2, height, width // 2 + 1)
            ffted = self.freq_conv(ffted)
            ffted = ffted.view(batch, channels, 2, height, width // 2 + 1)
            ffted = torch.complex(ffted[:, :, 0], ffted[:, :, 1])
            if self.highfreq_gate is not None:
                ffted = self.highfreq_gate(ffted) * ffted
            output = torch.fft.irfftn(ffted, s=(height, width), dim=(-2, -1), norm=self.fft_norm)
        return output.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self._fourier_branch(x)
        local = self.local(x)
        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        return x + gamma * self.fuse(torch.cat([local, freq], dim=1))
