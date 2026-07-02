# MIRFD-Net v2.3 Feature Statistics

This folder contains v2.3 per-sample feature statistics for all high-fusion ablation runs.

Raw CSV files contain one row per sample per stage. Summary CSV files aggregate by dataset and stage. The two additional summary files are:

- `v2_3_residual_gate_none_key_metrics.csv`: key stage-wise metrics for the best v2.3 variant on each dataset.
- `v2_3_variant_key_metrics.csv`: compact comparison among `high_raw_gate_none`, `residual_gate_none`, and `high_raw_gate_enhance`.

`pred_iou` and `pred_has_false_alarm` are sample-level final prediction metrics repeated on each stage row; they are not stage-specific predictions.
