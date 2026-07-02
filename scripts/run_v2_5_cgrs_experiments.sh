#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5"}
OUT_ROOT=${OUT_ROOT:-runs/v2_5_cgrs}
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
read -r -a GPUS <<< "${GPU_LIST}"

JOBS=(
  "nuaa_cgrs_unsupervised|configs/v2_5/mirfd_nuaa_sirst_ss2d_v2_5_cgrs_unsupervised.yaml"
  "nudt_cgrs_unsupervised|configs/v2_5/mirfd_nudt_sirst_ss2d_v2_5_cgrs_unsupervised.yaml"
  "irstd_cgrs_unsupervised|configs/v2_5/mirfd_irstd_1k_ss2d_v2_5_cgrs_unsupervised.yaml"
  "nuaa_cgrs_supervised|configs/v2_5/mirfd_nuaa_sirst_ss2d_v2_5_cgrs_supervised.yaml"
  "nudt_cgrs_supervised|configs/v2_5/mirfd_nudt_sirst_ss2d_v2_5_cgrs_supervised.yaml"
  "irstd_cgrs_supervised|configs/v2_5/mirfd_irstd_1k_ss2d_v2_5_cgrs_supervised.yaml"
)

for idx in "${!JOBS[@]}"; do
  IFS='|' read -r name config <<< "${JOBS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  out_dir="${OUT_ROOT}/${name}"
  log="${LOG_DIR}/${name}.log"
  (
    echo "[$(date '+%F %T')] gpu=${gpu} start ${name} config=${config}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON_BIN}" scripts/train.py \
      --config "${config}" \
      --output-dir "${out_dir}"
    echo "[$(date '+%F %T')] done ${name}"
  ) > "${log}" 2>&1 &
done

wait
echo "[$(date '+%F %T')] all v2.5 CGRS experiments done"
