#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

python train.py \
  --trial 710 \
  --agent_model transportmer \
  --seq_model sab \
  --macro_type poss_edge \
  --micro_type poss_edge \
  --node_in_dim 8 \
  --flip_pitch \
  --att_heads 4 \
  --coarse_dim 64 \
  --fine_dim 64 \
  --src_loss \
  --dst_loss \
  --macro_weight 1 \
  --rloss_weight 0 \
  --n_epochs 50 \
  --batch_size 32 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --seed 100 \
  --print_batch 200
