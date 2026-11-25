# The vLLM-Moet DeepSeek-V4.1-Flash sm_120 image with exact block top-k for the sparse indexer, switched on.
# BASE: a vLLM-Moet image built from https://github.com/kacper-daftcode/vLLM-Moet at commit 64096a4 (its sm120 Dockerfile).
# Build (no GPU needed): docker build -t vllm-moet-sm120:dsv41-64096a4-blocktopk .
ARG BASE=vllm-moet-sm120:dsv41-64096a4
FROM ${BASE}
COPY sm120_sparse_indexer /opt/vllm-moet/sm120_sparse_indexer/sm120_sparse_indexer
COPY patch/patched/model_executor/layers/sparse_attn_indexer.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
# Compile the CUDA extensions for sm_120a at build time (else they compile on first use, ~40 s per worker).
RUN cd /opt/vllm-moet/sm120_sparse_indexer && python3 -c "import sm120_sparse_indexer as sp; [print('built', so) for so in sp.build()]"
# Block top-k reproduces the dense path's deterministic selections, so it needs both switches below; prefill stays dense.
ENV VLLM_MOET_INDEXER_BLOCK_TOPK=1 \
    VLLM_MOET_DET_TOPK=1 \
    VLLM_MOET_CANDIDATE_DET_TOPK=1
LABEL org.opencontainers.image.title="vLLM-Moet DeepSeek-V4.1-Flash sm_120 + block top-k indexer" \
      org.opencontainers.image.source="https://github.com/GoedelMachines/sm120-block-topk" \
      org.opencontainers.image.licenses="Apache-2.0"
