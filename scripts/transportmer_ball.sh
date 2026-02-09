#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

python train.py \
  --trial 501 \
  --agent_model transportmer \
  --seq_model sab \
  --macro_type none \
  --micro_type ball \
  --node_in_dim 8 \
  --flip_pitch \
  --att_heads 4 \
  --coarse_dim 64 \
  --fine_dim 64 \
  --macro_weight 0 \
  --rloss_weight 0 \
  --n_epochs 50 \
  --batch_size 32 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --seed 100 \
  --print_batch 200
