#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

python train.py \
  --trial 600 \
  --agent_model set_tf \
  --seq_model bi_lstm \
  --macro_type none \
  --micro_type poss_edge \
  --node_in_dim 8 \
  --flip_pitch \
  --seq_dim 128 \
  --src_loss \
  --dst_loss \
  --macro_weight 0 \
  --rloss_weight 0 \
  --n_epochs 50 \
  --batch_size 32 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --seed 100 \
  --print_batch 200