#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

python evaluate.py \
  --trials 150 156 \
  --eval_source kleague \
  --file_start 12 \
  --file_count 3 \
  --model_file state_dict_best_acc.pt