#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_LIST=${GPU_LIST:-"0 1 2"}
OUT_DIR=${OUT_DIR:-docs/diagnostics/branch_probe/v2_5}
LOG_DIR=${LOG_DIR:-runs/v2_5_branch_probe/logs}
EPOCHS=${EPOCHS:-30}
STAGES=${STAGES:-"1,2,3"}
BRANCHES=${BRANCHES:-"low,residual,high_raw,low_residual,low_high_raw,low_residual_high_raw"}

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
read -r -a GPUS <<< "${GPU_LIST}"

JOBS=(
  "nuaa|NUAA-SIRST|configs/mirfd_nuaa_sirst_ss2d_v2_3_block_residual_gate_none.yaml|runs/v2_3_ablation/nuaa_block_residual_gate_none/best_iou.pt"
  "nudt|NUDT-SIRST|configs/mirfd_nudt_sirst_ss2d_v2_3_block_residual_gate_none.yaml|runs/v2_3_ablation/nudt_block_residual_gate_none/best_iou.pt"
  "irstd|IRSTD-1K|configs/mirfd_irstd_1k_ss2d_v2_3_block_residual_gate_none.yaml|runs/v2_3_ablation/irstd_block_residual_gate_none/best_iou.pt"
)

for idx in "${!JOBS[@]}"; do
  IFS='|' read -r key dataset config ckpt <<< "${JOBS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  log="${LOG_DIR}/${key}.log"
  csv="${OUT_DIR}/${key}_branch_probe.csv"
  (
    echo "[$(date '+%F %T')] gpu=${gpu} start ${key} branch probe"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON_BIN}" scripts/train_branch_probe.py \
      --config "${config}" \
      --checkpoint "${ckpt}" \
      --dataset-name "${dataset}" \
      --output-csv "${csv}" \
      --stage "${STAGES}" \
      --branch "${BRANCHES}" \
      --epochs "${EPOCHS}"
    echo "[$(date '+%F %T')] done ${key} branch probe"
  ) > "${log}" 2>&1 &
done

wait
echo "[$(date '+%F %T')] all v2.5 branch probes done"
