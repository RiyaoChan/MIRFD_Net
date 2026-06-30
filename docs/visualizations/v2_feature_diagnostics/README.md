# V2 Feature Diagnostic Heatmap Samples

This folder contains 20 test-set diagnostic samples for each dataset:

- `nuaa/`: NUAA-SIRST
- `nudt/`: NUDT-SIRST
- `irstd/`: IRSTD-1K

Each image visualizes the V2 feature path on the same sample. The first row shows input, ground truth, and V2 prediction. The following rows contain two diagnostic groups.

Structure diagnostics for stage-2, stage-3, and stage-4:

- `low0`: raw Mamba/SS2D-induced low-frequency semantic approximation before low smoothing.
- `low`: Mamba/SS2D-induced low-frequency semantic approximation.
- `residual`: raw `F - low` residual.
- `high_raw`: high branch output before gate modulation.
- `gate`: target-aware gate response.
- `high_hat`: gated high response used by the decoder.

Frequency diagnostics for stage-2, stage-3, and stage-4:

- `FFT(low)`
- `FFT(residual)`
- `FFT(high_raw)`
- `FFT(high_hat)`

The `low0`, `low`, `residual`, `high_raw`, and `high_hat` structure heatmaps use channel-wise `abs().mean(dim=0)`. The `gate` heatmap uses channel-wise `mean(dim=0)`. FFT heatmaps use channel-averaged `log(1 + |fftshift(fft2(feature))|)`.
