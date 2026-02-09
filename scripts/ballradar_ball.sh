#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

python train.py \
  --trial 665 \
  --agent_model set_tf \
  --seq_model sab \
  --macro_type poss_edge \
  --micro_type ball \
  --node_in_dim 8 \
  --flip_pitch \
  --seq_dim 128 \
  --macro_weight 20 \
  --rloss_weight 1 \
  --n_epochs 50 \
  --batch_size 16 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --seed 100 \
  --print_batch 200