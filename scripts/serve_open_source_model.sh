#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to a local Hugging Face checkpoint}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

extra_args=()
if [[ "${SKIP_MM_PROFILING}" == "1" ]]; then
  extra_args+=(--skip-mm-profiling)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  extra_args+=(--enforce-eager)
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --trust-remote-code \
  "${extra_args[@]}"
