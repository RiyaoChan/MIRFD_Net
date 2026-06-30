# V2 Feature Diagnostic Heatmap Samples

This folder contains 20 test-set diagnostic samples for each dataset:

- `nuaa/`: NUAA-SIRST
- `nudt/`: NUDT-SIRST
- `irstd/`: IRSTD-1K

Each image visualizes the V2 feature path on the same sample. The first row shows input, ground truth, and V2 prediction. The following rows show stage-2, stage-3, and stage-4 components:

- `low`: Mamba/SS2D-induced low-frequency semantic approximation.
- `residual`: raw `F - low` residual.
- `high_raw`: high branch output before gate modulation.
- `gate`: target-aware gate response.
- `high_hat`: gated high response used by the decoder.

The `low`, `residual`, `high_raw`, and `high_hat` heatmaps use channel-wise `abs().mean(dim=0)`. The `gate` heatmap uses channel-wise `mean(dim=0)`.
