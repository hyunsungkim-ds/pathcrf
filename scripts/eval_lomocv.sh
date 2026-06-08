#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
  --trials 150 156 \
  --eval_source sportec \
  --model_file state_dict_best_acc.pt