# MIRFD-Net v2.3 Residual Fusion Feature Diagnostics

This folder contains 20 test-set diagnostic samples for each dataset using the best v2.3 variant:

- `block_fusion_high_source: residual`
- `gate_mode: none`
- `decoder_high_source: high_raw`

Each sample contains:

- Input, ground truth, prediction probability, and thresholded prediction mask.
- Stage-1 low/residual/high responses and FFT maps.
- Stage-2/3/4 `low0`, `low`, `residual`, `high_raw`, `high_hat`, `high_for_fusion`, and `gate`.
- FFT maps for `low`, `residual`, `high_raw`, `high_hat`, and `high_for_fusion`.

For this variant, `high_for_fusion` is the feature actually used by the MIRFD Block fusion path. It is intentionally `residual`, while `high_raw` and `high_hat` are retained for diagnosis.
