#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

RUNS=(
  "configs/mirfd_nuaa_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml|runs/v2_2_ablation/nuaa_stage1_identity_stage2_fsre"
  "configs/mirfd_nudt_sirst_ss2d_v2_2_stage1_identity_stage2_fsre.yaml|runs/v2_2_ablation/nudt_stage1_identity_stage2_fsre"
  "configs/mirfd_irstd_1k_ss2d_v2_2_stage1_identity_stage2_fsre.yaml|runs/v2_2_ablation/irstd_stage1_identity_stage2_fsre"
)

mkdir -p runs/v2_2_ablation/logs

for index in "${!RUNS[@]}"; do
  IFS='|' read -r config output_dir <<< "${RUNS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  run_name="${output_dir#runs/v2_2_ablation/}"
  log_path="runs/v2_2_ablation/logs/${run_name}.log"
  echo "Launching ${run_name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train.py \
    --config "${config}" \
    --output-dir "${output_dir}" \
    > "${log_path}" 2>&1 &
done

wait
