#!/usr/bin/env bash
set -euo pipefail

# Example: 2-GPU single-node training via DDP (recommended)
# - batch_size is PER GPU (global batch = batch_size * nproc_per_node)
# - set CUDA_VISIBLE_DEVICES to the GPUs you want to use

export CUDA_VISIBLE_DEVICES=2,3

torchrun --standalone --nproc_per_node=2 train.py \
  --trial 101 \
  --agent_model gat \
  --seq_model lstm \
  --macro_type poss_next \
  --micro_type ball \
  --node_in_dim 8 \
  --edge_in_dim 2 \
  --macro_weight 20 \
  --n_epochs 100 \
  --batch_size 64 \
  --start_lr 1e-3 \
  --min_lr 1e-5 \
  --seed 100

