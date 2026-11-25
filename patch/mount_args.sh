#!/usr/bin/env bash
# Prints docker -v arguments that overlay the four pre-patched vLLM files (patch/patched/, made by
# patch_vllm_sparse_indexer_sm120.py from the image's own copies) and mount this repo at /opt/vllm-moet/sm120_sparse_indexer.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
for f in utils/deep_gemm.py model_executor/layers/sparse_mqa_indexer.py v1/attention/backends/mla/sparse_indexer.py \
         model_executor/kernels/attention/dsa/sparse_mqa_logits.py; do
  printf -- '-v %s:%s:ro ' "$HERE/patch/patched/$f" "$VLLM/$f"
done
printf -- '-v %s:/opt/vllm-moet/sm120_sparse_indexer:ro\n' "$HERE"
