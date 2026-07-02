#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
ROOT_DIR="${ROOT_DIR:-/DATA20T/bip/cry/code/MIRFD_Net}"
GPU_LIST="${GPU_LIST:-0 1 2}"

cd "${ROOT_DIR}"
mkdir -p runs/v2_4_ffc_ablation/logs

JOBS=(
  "configs/mirfd_nuaa_sirst_ss2d_v2_4_ffc_residual_gate_none.yaml|runs/v2_4_ffc_ablation/nuaa_ffc_residual_gate_none"
  "configs/mirfd_nudt_sirst_ss2d_v2_4_ffc_residual_gate_none.yaml|runs/v2_4_ffc_ablation/nudt_ffc_residual_gate_none"
  "configs/mirfd_irstd_1k_ss2d_v2_4_ffc_residual_gate_none.yaml|runs/v2_4_ffc_ablation/irstd_ffc_residual_gate_none"
)

run_job() {
  local gpu="$1"
  local job_index="$2"
  local spec="${JOBS[$job_index]}"
  local config="${spec%%|*}"
  local output_dir="${spec##*|}"
  local run_name="${output_dir#runs/v2_4_ffc_ablation/}"
  local log_path="runs/v2_4_ffc_ablation/logs/${run_name}.log"

  mkdir -p "${output_dir}"
  echo "[$(date '+%F %T')] gpu=${gpu} start ${run_name} config=${config}" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train.py \
    --config "${config}" \
    --output-dir "${output_dir}" \
    >> "${log_path}" 2>&1
  echo "[$(date '+%F %T')] gpu=${gpu} done ${run_name}" | tee -a "${log_path}"
}

read -r -a GPUS <<< "${GPU_LIST}"
for job_index in "${!JOBS[@]}"; do
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  run_job "${gpu}" "${job_index}" &
done

wait
echo "[$(date '+%F %T')] all v2.4 FFC experiments finished"
