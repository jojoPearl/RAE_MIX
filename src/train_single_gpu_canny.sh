#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled

export CUDA_VISIBLE_DEVICES=0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-easycontrol_full}"
export ENTITY="${ENTITY:-claraxu}"
export ENABLE_LOG_TXT="${ENABLE_LOG_TXT:-0}"
export SAVE_WORKTREE="${SAVE_WORKTREE:-0}"

DATA_PATH="/home/bjia-25/workspace/papers/RAE/datasets/imagenet/train"
RESULT_DIR="ckpts/easycontrol"

CONFIG_STAGE1="configs/stage2/training/ImageNet256/easycontrol_stage1.yaml"
CONFIG_STAGE2="configs/stage2/training/ImageNet256/easycontrol_stage2.yaml"
CONFIG_STAGE3="configs/stage2/training/ImageNet256/easycontrol_stage3.yaml"

CKPT_DIR="${RESULT_DIR}/${EXPERIMENT_NAME}/checkpoints"
LOG_DIR="${RESULT_DIR}/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/log_$(date +%Y%m%d_%H%M%S).log}"

FORCE_FRESH="${FORCE_FRESH:-0}"
NOHUP="${NOHUP:-1}"

STAGE1_END=10000
STAGE2_END=30000
STAGE3_END=150000

echo "======================================"
echo "EasyControl chained stages"
echo "Experiment: $EXPERIMENT_NAME"
echo "CKPT_DIR: $CKPT_DIR"
echo "FORCE_FRESH: $FORCE_FRESH"
echo "NOHUP: $NOHUP"
echo "Stage ends: s1=$STAGE1_END s2=$STAGE2_END s3=$STAGE3_END"
echo "======================================"

detect_latest_ckpt() {
  local latest=""
  if [ -d "$CKPT_DIR" ]; then
    latest="$(ls -1 "$CKPT_DIR"/adapter_step-*.pt 2>/dev/null | sort | tail -n 1 || true)"
  fi
  echo "$latest"
}

parse_step_from_ckpt() {
  local ckpt_path="$1"
  local step=""
  step="$(basename "$ckpt_path" | sed -n 's/^adapter_step-\([0-9]\+\)\.pt$/\1/p' | sed 's/^0*//')"
  if [ -z "$step" ]; then
    step="0"
  fi
  echo "$step"
}

move_as_backup_if_fresh() {
  if [ "$FORCE_FRESH" = "1" ]; then
    if [ -d "${RESULT_DIR}/${EXPERIMENT_NAME}" ]; then
      local ts
      ts="$(date +%Y%m%d_%H%M%S)"
      local backup_dir="${RESULT_DIR}/${EXPERIMENT_NAME}_backup_${ts}"
      echo "FORCE_FRESH=1 -> moving existing dir to: $backup_dir"
      mv "${RESULT_DIR}/${EXPERIMENT_NAME}" "$backup_dir"
    fi
  fi
}

select_stage_by_step() {
  local step="$1"
  if [ "$step" -ge "$STAGE3_END" ]; then
    echo "done"
  elif [ "$step" -ge "$STAGE2_END" ]; then
    echo "stage3"
  elif [ "$step" -ge "$STAGE1_END" ]; then
    echo "stage2"
  else
    echo "stage1"
  fi
}

config_for_stage() {
  local stage="$1"
  if [ "$stage" = "stage1" ]; then
    echo "$CONFIG_STAGE1"
  elif [ "$stage" = "stage2" ]; then
    echo "$CONFIG_STAGE2"
  elif [ "$stage" = "stage3" ]; then
    echo "$CONFIG_STAGE3"
  else
    echo ""
  fi
}

run_one_stage() {
  local stage="$1"
  local cfg="$2"
  local resume_ckpt="$3"

  if [ ! -f "$cfg" ]; then
    echo "ERROR: config not found: $cfg"
    exit 1
  fi

  local cmd=(python -u src/train_easycontrol.py
    --config "$cfg"
    --data-path "$DATA_PATH"
    --results-dir "$RESULT_DIR"
    --image-size 256
    --precision fp32
  )

  if [ -n "$resume_ckpt" ]; then
    if [ ! -f "$resume_ckpt" ]; then
      echo "ERROR: resume checkpoint not found: $resume_ckpt"
      exit 1
    fi
    cmd+=(--resume "$resume_ckpt")
  fi

  echo "--------------------------------------"
  echo "Running $stage"
  echo "Config: $cfg"
  if [ -n "$resume_ckpt" ]; then
    echo "Resume: $resume_ckpt"
  else
    echo "Resume: (none)"
  fi
  echo "Log: $RUN_LOG"
  echo "Command:"
  printf '  %q' "${cmd[@]}"
  echo
  echo "--------------------------------------"

  "${cmd[@]}"
}

controller_main() {
  move_as_backup_if_fresh

  while true; do
    local latest_ckpt
    local cur_step
    local selected_stage
    local selected_cfg
    local resume_ckpt

    latest_ckpt="$(detect_latest_ckpt)"
    cur_step="0"
    resume_ckpt=""

    if [ "$FORCE_FRESH" = "1" ]; then
      latest_ckpt=""
      cur_step="0"
      resume_ckpt=""
    else
      if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt" ]; then
        cur_step="$(parse_step_from_ckpt "$latest_ckpt")"
        resume_ckpt="$latest_ckpt"
      fi
    fi

    echo "======================================"
    echo "Latest ckpt: ${latest_ckpt:-<none>}"
    echo "CUR_STEP: $cur_step"
    echo "======================================"

    selected_stage="$(select_stage_by_step "$cur_step")"
    if [ "$selected_stage" = "done" ]; then
      echo "Reached stage3 end ($STAGE3_END). Exit."
      break
    fi

    selected_cfg="$(config_for_stage "$selected_stage")"
    if [ -z "$selected_cfg" ]; then
      echo "ERROR: failed to map stage to config: $selected_stage"
      exit 1
    fi

    run_one_stage "$selected_stage" "$selected_cfg" "$resume_ckpt"

    FORCE_FRESH="0"
  done

  echo "All stages completed."
}

if [ "$NOHUP" = "1" ] && [ "${RUNNING_IN_NOHUP:-0}" != "1" ]; then
  export RUNNING_IN_NOHUP=1
  export RUN_LOG
  echo "Launching controller in background (nohup)."
  echo "Unified log: $RUN_LOG"
  nohup bash "$0" "$@" > "$RUN_LOG" 2>&1 &
  echo "Controller PID: $!"
  echo "Tail with: tail -f $RUN_LOG"
  exit 0
fi

if [ "${RUNNING_IN_NOHUP:-0}" != "1" ]; then
  exec > >(tee -a "$RUN_LOG") 2>&1
fi
echo "Unified log: $RUN_LOG"

controller_main

############################################################
# ---------------- ORIGINAL COMMANDS (COMMENTED) ----------
############################################################

# python src/sample.py \
#   --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
#   --seed 42

# nohup
# CUDA_VISIBLE_DEVICES=0 python -u src/train_single_gpu.py \
#   --config /home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_test.yaml \
#   --data-path /home/bjia-25/workspace/papers/RAE/datasets/imagenet/train \
#   --results-dir ckpts/debug_canny_full_label \
#   --compile \
#   --image-size 256 --precision fp32 \
#   > ckpts/training_log.out 2>&1 &
#
# nohup
# CUDA_VISIBLE_DEVICES=0 
# python src/train_easycontrol.py \
#   --config /home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_test.yaml \
#   --data-path /home/bjia-25/workspace/papers/RAE/datasets/imagenet/train \
#   --results-dir ckpts/easycontrol \
#   --image-size 256 \
#   --precision fp32
#  > ckpts/training_log.out 2>&1 &
#
# echo "Training started in background. PID: $!"

# nohup python src/sample_dump.py \
#   --config /home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-train_easycontrol.yaml \
#   --data-path /home/bjia-25/workspace/papers/RAE/datasets/imagenet/val \
#   --image-size 256 \
#   --precision bf16 \
#   --num-samples 10000 \
#   --batch-size 2 \
#   --outdir eval_10k  > training_log.out 2>&1 &


# 10k
# python RAE_MIX/src/sample_dump.py \
#   --config /path/to/config.yaml \
#   --data-path /path/to/val \
#   --adapter-ckpt /path/to/adapter.pt \
#   --num-samples 10000 \
#   --batch-size 4 \
#   --outdir out_dump_10k

# # offline eval (control only)
# python RAE_MIX/src/eval_from_folders.py --root out_dump_10k

# # offline eval (baseline + control)
# nohup python src/eval_from_folders.py --root eval_10k --compare-baseline --batch-size 64 --use-is --is-splits 10  > eval_10k/eval_log.out 2>&1 &



# python src/sample_easycontrol.py \
#   --config /home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-train_easycontrol.yaml \
#   --data-path /path/to/val \
#   --canny-path /home/bjia-25/workspace/papers/RAE/code/rae_project/transformer-imagenet-ctrl/out_ctrl/g_canny.png \
#   --class-label 207 \
#   --adapter-ckpt /home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/ckpts/easycontrol/easycontrol_full/checkpoints/adapter_step-0110000.pt \
#   --num-samples 4 \
#   --control-scale 3.2 \
#   --canny-noise-std 0.05 \
#   --outdir /home/bjia-25/workspace/papers/RAE/code/rae_project/RAE_MIX/ckpts/out \
#   --save-individual



# python src/sample_easycontrol_avg.py \
#   --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-train_easycontrol.yaml \
#   --data-path /home/bjia-25/workspace/papers/RAE/datasets/imagenet/val \
#   --outdir ckpts/easycontrol/feature_only_run \
#   --target-class-id 207 \
#   --image-size 256 \
#   --precision bf16 \
#   --adapter-ckpt ckpts/easycontrol/easycontrol_full/checkpoints/adapter_step-0070000.pt \
#   --num-samples 16 \
#   --seed 42 \
#   --steps 50 \
#   --sampler heun \
#   --control-scale 3.0 \
#   --repa-lambda 0.8 \
#   --repa-loss cosine \
#   --guidance-low 0.0 \
#   --guidance-high 0.7 \
#   --avgfeat-only \
#   --save-individual
