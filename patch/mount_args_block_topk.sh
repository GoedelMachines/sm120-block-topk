#!/usr/bin/env bash
# Prints docker arguments for VLLM_MOET_INDEXER_BLOCK_TOPK=1: the pre-patched sparse_attn_indexer.py (patch/patched/,
# made by patch_vllm_indexer_block_topk_sm120.py from the image's own copy) over the image's, this repo at
# /opt/vllm-moet/sm120_sparse_indexer (the package + the prebuilt extensions in build/), and the switch.
# Usage: EXTRA_DOCKER_ARGS="$(patch/mount_args_block_topk.sh)" SERVED=1 ... docker/sm120/run-dsv41.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
f=model_executor/layers/sparse_attn_indexer.py
printf -- '-v %s:%s:ro ' "$HERE/patch/patched/$f" "$VLLM/$f"
printf -- '-v %s:/opt/vllm-moet/sm120_sparse_indexer:ro ' "$HERE"
printf -- '-e VLLM_MOET_INDEXER_BLOCK_TOPK=1\n'
