#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

python train.py \
  --trial 750 \
  --agent_model transportmer \
  --seq_model sab \
  --crf_model dynamic_sparse_crf \
  --macro_type poss_edge \
  --micro_type poss_edge \
  --node_in_dim 8 \
  --flip_pitch \
  --att_heads 4 \
  --coarse_dim 64 \
  --fine_dim 64 \
  --crf_edge_dim 16 \
  --src_loss \
  --dst_loss \
  --ce_epochs 0 \
  --macro_weight 1 \
  --rloss_weight 0 \
  --crf_weight 1 \
  --n_epochs 50 \
  --batch_size 32 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --seed 100 \
  --print_batch 200
