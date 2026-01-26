#!/bin/bash
set -e

export IMAGENET_DIR=/home/bjia-25/workspace/papers/RAE/datasets/imagenet
export CKPT_DIR=/home/bjia-25/workspace/papers/RAE/code/rae_project/transformer-imagenet-ctrl/outputs
export INIT_WTS=/home/bjia-25/workspace/papers/RAE/code/rae_project/transformer-imagenet-ctrl/checkpoints/SiT-XL-2-256-REPA.pt

mkdir -p ${CKPT_DIR}/sit_canny

echo "==== Training SiT with Canny (quick run) ===="

torchrun --nproc_per_node=1 train.py \
  --base_model sit \
  --ctrl canny \
  --data_path "${IMAGENET_DIR}" \
  --local_out_dir_path "${CKPT_DIR}/sit_canny" \
  --init_wts "${INIT_WTS}" \
  --bs 16 \
  --ep 1 \
  --cond_patches max

echo "==== Generating ONE image with Canny control ===="

torchrun --nproc_per_node=1 eval.py \
  --base_model sit \
  --ctrl canny \
  --eval_ctrl canny \
  --data_path "${IMAGENET_DIR}" \
  --local_out_dir_path "${CKPT_DIR}/sit_canny" \
  --init_wts "${INIT_WTS}" \
  --ngen 1 \
  --bs 1 \
  --eval_ckpts last

echo "==== DONE ===="
echo "Check results in: ${CKPT_DIR}/sit_canny"
