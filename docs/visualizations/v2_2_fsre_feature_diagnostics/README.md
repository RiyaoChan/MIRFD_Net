# MIRFD-Net v2.2 FSRE Feature Diagnostics

This folder contains test-set diagnostic visualizations for the first v2.2 FSRE run:

- `nuaa_v2_diagnostic/`
- `nudt_v2_diagnostic/`
- `irstd_v2_diagnostic/`

Each dataset folder contains 8 samples and a `contact_sheet.png`.

Each sample visualizes:

- Input, GT, and final prediction.
- Structure maps for stage-2/3/4: `low0`, `low`, `residual`, `high_raw`, `gate`, `high_hat`.
- FFT maps for stage-2/3/4: `FFT(low)`, `FFT(residual)`, `FFT(high_raw)`, `FFT(high_hat)`.

For v2.2, `high_raw` is the output of the selected high enhancer. In the `stage1_identity_stage2_fsre` run, stage-1 uses identity residual while MIRFD stages use `FrequencySelectiveResidualEnhancer`; decoder high skip uses `high_raw` for enabled stages `[1, 2]`.

The corresponding quantitative CSV files are in `docs/diagnostics/feature_statistics/*v2_2_stage1_identity_stage2_fsre.csv`.
