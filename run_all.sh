#!/usr/bin/env bash
# Every check of this repo on one GPU (set GPU=<index or uuid>), logs under results/. ~12 min; the last bench
# needs 5 GiB of GPU memory, the others stay under 0.8 GiB.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
V=/usr/local/lib/python3.12/dist-packages/vllm
run() { local log="results/$1.log"; shift; echo "== $* -> $log"; "$@" > "$log" 2>&1; echo "EXIT=$?" >> "$log"; tail -2 "$log"; }
run test_block_topk_full ./run_in_image.sh python3 tests/test_block_topk.py
EXTRA_DOCKER_ARGS="-v $PWD/patch/patched/model_executor/layers/sparse_attn_indexer.py:$V/model_executor/layers/sparse_attn_indexer.py:ro -e VLLM_MOET_DET_TOPK=1 -e VLLM_MOET_SPARSE_INDEXER_DIR=$PWD" \
  run test_vllm_block_patch ./run_in_image.sh python3 tests/test_vllm_block_patch.py
run test_exact_full ./run_in_image.sh python3 tests/test_exact.py --cases ref,decode,prefill,source,sbk16
EXTRA_DOCKER_ARGS="$(patch/mount_args.sh) -e VLLM_MOET_SPARSE_INDEXER=1 -e VLLM_MOET_SPARSE_INDEXER_DIR=$PWD" \
  run test_vllm_patch ./run_in_image.sh python3 tests/test_vllm_patch.py
ALLOC_CONF= run graph_mem ./run_in_image.sh python3 tests/graph_mem.py
run bench_block_topk ./run_in_image.sh python3 bench/bench_block_topk.py --rows 8,128,384,512 \
  --ctx 4096,16384,65536,262144,1048576 --parts --dense-chunk-mib 512 --iters 30
# more K than the L2 holds: 64 requests x 1M tokens, every request its own pages (4.4 GiB of K, 5 GiB peak)
run bench_block_topk_dram ./run_in_image.sh python3 bench/bench_block_topk.py --kind source,ratio2 --rows 128,512 \
  --ctx 262144,1048576 --distinct --pool-mib 4500 --parts --dense-chunk-mib 512 --iters 10
echo ALL DONE
