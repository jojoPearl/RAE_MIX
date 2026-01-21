#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="src/eval/infer.py"

BASE_IMAGE="assets/out_ctrl/r.png"
CANNY_IMG="assets/out_ctrl/r_canny.png"
DEPTH_IMG="assets/out_ctrl/r_depth.png"

LOG_DIR="assets/out_ctrl/logs"
mkdir -p "${LOG_DIR}"

# ===== 可改超参 =====
CLASS_ID="${CLASS_ID:-207}"          # 207 golden retriever
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-50}"

LR_ADAPTER="${LR_ADAPTER:-1e-4}"
CONTROL_WEIGHT="${CONTROL_WEIGHT:-1.0}"
SMOOTH_W="${SMOOTH_W:-0.02}"

INFER_STEPS="${INFER_STEPS:-64}"
INFER_CONTROL_WEIGHT="${INFER_CONTROL_WEIGHT:-0.6}"
GAMMA_C="${GAMMA_C:-1.2}"
IMG2IMG_STRENGTH="${IMG2IMG_STRENGTH:-0.6}"  # 0.2~0.4 更像原图

# LoRA（默认不开；要开：export USE_LORA=1）
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

use_lora_flag=""
if [[ "${USE_LORA}" == "1" ]]; then
  use_lora_flag="--use_lora --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA}"
fi

TS="$(date +%Y%m%d_%H%M%S)"

echo "=============================="
echo "RUN ALL   TS=${TS}"
echo "CLASS_ID=${CLASS_ID}"
echo "USE_LORA=${USE_LORA}"
echo "=============================="

# -----------------------------
# 1) TRAIN: 只训练三次（canny/depth/both）
# -----------------------------
for CTRL_MODE in canny depth both; do
  LOG_FILE="${LOG_DIR}/TRAIN_${CTRL_MODE}_${TS}.log"
  echo
  echo ">>> TRAIN CTRL_MODE=${CTRL_MODE}"
  echo ">>> LOG: ${LOG_FILE}"

  "${PYTHON_BIN}" "${SCRIPT}" \
    --mode train \
    --ctrl_mode "${CTRL_MODE}" \
    --base_image "${BASE_IMAGE}" \
    --canny "${CANNY_IMG}" \
    --depth "${DEPTH_IMG}" \
    --class_id "${CLASS_ID}" \
    --steps "${TRAIN_STEPS}" \
    --save_every "${SAVE_EVERY}" \
    --log_every "${LOG_EVERY}" \
    --lr_adapter "${LR_ADAPTER}" \
    --control_weight "${CONTROL_WEIGHT}" \
    --smooth_w "${SMOOTH_W}" \
    ${use_lora_flag} \
    2>&1 | tee "${LOG_FILE}"
done

# -----------------------------
# 2) INFER: 每个 ctrl_mode 跑 img2img + txt2img
# -----------------------------
for CTRL_MODE in canny depth both; do
  for INFER_MODE in img2img txt2img; do
    LOG_FILE="${LOG_DIR}/INFER_${CTRL_MODE}_${INFER_MODE}_${TS}.log"
    echo
    echo ">>> INFER CTRL_MODE=${CTRL_MODE}  INFER_MODE=${INFER_MODE}"
    echo ">>> LOG: ${LOG_FILE}"

    "${PYTHON_BIN}" "${SCRIPT}" \
      --mode infer \
      --ctrl_mode "${CTRL_MODE}" \
      --infer_mode "${INFER_MODE}" \
      --base_image "${BASE_IMAGE}" \
      --canny "${CANNY_IMG}" \
      --depth "${DEPTH_IMG}" \
      --class_id "${CLASS_ID}" \
      --infer_steps "${INFER_STEPS}" \
      --infer_control_weight "${INFER_CONTROL_WEIGHT}" \
      --gamma_c "${GAMMA_C}" \
      --img2img_strength "${IMG2IMG_STRENGTH}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      2>&1 | tee "${LOG_FILE}"
  done
done

echo
echo "=============================="
echo "ALL DONE."
echo "train: assets/out_ctrl/train/adapter/{canny|depth|both}/"
echo "infer: assets/out_ctrl/infer/{canny|depth|both}/{img2img|txt2img}/"
echo "logs : assets/out_ctrl/logs/"
echo "=============================="
