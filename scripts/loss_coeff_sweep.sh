#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

python train.py \
  --trial 196 \
  --sources sportec \
  --lomocv \
  --cv_fold_idx 6 \
  --agent_model set_tf \
  --seq_model sab \
  --crf_model dynamic_sparse_crf \
  --macro_type poss_edge \
  --micro_type poss_edge \
  --node_in_dim 8 \
  --target_fps 5 \
  --window_stride 1 \
  --flip_pitch \
  --seq_dim 128 \
  --crf_edge_dim 16 \
  --emit_loss \
  --src_loss \
  --dst_loss \
  --ce_epochs 0 \
  --macro_weight 0.25 \
  --micro_weight 0.5 \
  --rloss_weight 0 \
  --crf_weight 1 \
  --n_epochs 50 \
  --batch_size 32 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --seed 100 \
  --print_batch 200