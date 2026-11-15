#!/usr/bin/env bash
# 起服务：8GB 显存的 4060 上跑 Qwen2.5-1.5B-Instruct，留出 KV cache 的余量
set -e
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
vllm serve "$MODEL" \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --disable-log-requests
