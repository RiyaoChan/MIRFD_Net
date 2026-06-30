#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
ROOT_DIR="${ROOT_DIR:-/DATA20T/bip/cry/code/MIRFD_Net}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6}"

cd "${ROOT_DIR}"
mkdir -p runs/v2_1_ablation/logs

JOBS=(
  "configs/mirfd_nuaa_sirst_ss2d_v2_1_centered_shallow.yaml|runs/v2_1_ablation/nuaa_centered_shallow"
  "configs/mirfd_nudt_sirst_ss2d_v2_1_centered_shallow.yaml|runs/v2_1_ablation/nudt_centered_shallow"
  "configs/mirfd_irstd_1k_ss2d_v2_1_centered_shallow.yaml|runs/v2_1_ablation/irstd_centered_shallow"
  "configs/mirfd_nuaa_sirst_ss2d_v2_1_centered_shallow_add_scaled.yaml|runs/v2_1_ablation/nuaa_centered_shallow_add_scaled"
  "configs/mirfd_nudt_sirst_ss2d_v2_1_centered_shallow_add_scaled.yaml|runs/v2_1_ablation/nudt_centered_shallow_add_scaled"
  "configs/mirfd_irstd_1k_ss2d_v2_1_centered_shallow_add_scaled.yaml|runs/v2_1_ablation/irstd_centered_shallow_add_scaled"
  "configs/mirfd_nuaa_sirst_ss2d_v2_1_centered_gate.yaml|runs/v2_1_ablation/nuaa_centered_gate"
  "configs/mirfd_nudt_sirst_ss2d_v2_1_centered_gate.yaml|runs/v2_1_ablation/nudt_centered_gate"
  "configs/mirfd_irstd_1k_ss2d_v2_1_centered_gate.yaml|runs/v2_1_ablation/irstd_centered_gate"
  "configs/mirfd_nuaa_sirst_ss2d_v2_1_shallow_high_skip.yaml|runs/v2_1_ablation/nuaa_shallow_high_skip"
  "configs/mirfd_nudt_sirst_ss2d_v2_1_shallow_high_skip.yaml|runs/v2_1_ablation/nudt_shallow_high_skip"
  "configs/mirfd_irstd_1k_ss2d_v2_1_shallow_high_skip.yaml|runs/v2_1_ablation/irstd_shallow_high_skip"
)

run_job() {
  local gpu="$1"
  local job_index="$2"
  local spec="${JOBS[$job_index]}"
  local config="${spec%%|*}"
  local output_dir="${spec##*|}"
  local run_name="${output_dir#runs/v2_1_ablation/}"
  local log_path="runs/v2_1_ablation/logs/${run_name}.log"

  mkdir -p "${output_dir}"
  echo "[$(date '+%F %T')] gpu=${gpu} start ${run_name} config=${config}" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train.py \
    --config "${config}" \
    --output-dir "${output_dir}" \
    >> "${log_path}" 2>&1
  echo "[$(date '+%F %T')] gpu=${gpu} done ${run_name}" | tee -a "${log_path}"
}

run_worker() {
  local gpu="$1"
  shift
  for job_index in "$@"; do
    run_job "${gpu}" "${job_index}"
  done
}

read -r -a GPUS <<< "${GPU_LIST}"
for worker_index in "${!GPUS[@]}"; do
  assigned=()
  job_index="${worker_index}"
  while [ "${job_index}" -lt "${#JOBS[@]}" ]; do
    assigned+=("${job_index}")
    job_index=$((job_index + ${#GPUS[@]}))
  done
  if [ "${#assigned[@]}" -gt 0 ]; then
    run_worker "${GPUS[$worker_index]}" "${assigned[@]}" &
  fi
done

wait
echo "[$(date '+%F %T')] all v2.1 experiments finished"
