#!/bin/bash
set -e

echo "=== entrypoint.sh starting, vllm version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1) ==="
echo "=== CUDA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1) ==="

python3 -m vllm.entrypoints.openai.api_server \
  --model "${HF_MODEL_ID}" \
  --tensor-parallel-size "${SM_NUM_GPUS:-8}" \
  --port 8000 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --trust-remote-code &

exec python3 /opt/serve.py
