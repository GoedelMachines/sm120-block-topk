#!/usr/bin/env bash
# Run a command from this directory inside the vllm-moet image on ONE GPU (by UUID), as the calling user, with
# writable caches under ./cache. Usage: GPU=GPU-<uuid> ./run_in_image.sh python3 tests/test_exact.py --quick
# GPU=none attaches no GPU (builds: ./build.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-vllm-moet-sm120:dsv41-64096a4}"  # vLLM-Moet image built at commit 64096a4 (see README)
GPU="${GPU:?set GPU=<index or GPU-uuid of one sm_120 GPU>, or GPU=none}"
NAME="${NAME:-sm120-sparse-indexer-$$}"
mkdir -p "$HERE/cache"
GPU_ARGS=(--gpus "\"device=$GPU\"")
[ "$GPU" = none ] && GPU_ARGS=()
exec docker run --rm --name "$NAME" "${GPU_ARGS[@]}" --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e TORCH_EXTENSIONS_DIR="$HERE/cache/torch_ext" -e VLLM_CACHE_ROOT="$HERE/cache/vllm" \
  -e TRITON_CACHE_DIR="$HERE/cache/triton" -e DG_JIT_CACHE_DIR="$HERE/cache/deep_gemm" \
  -e VLLM_MOET_DET_TOPK_DIR=/opt/vllm-moet/dsv41_sm120/det_topk -e VLLM_MOET_CANDIDATE_DET_TOPK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF-expandable_segments:True}" \
  ${EXTRA_DOCKER_ARGS:-} \
  -v "$HERE:$HERE" -w "$HERE" --entrypoint "$1" "$IMAGE" "${@:2}"
