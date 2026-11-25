#!/usr/bin/env python3
"""The patched vLLM entry points on sm_120 against the dense path (run in a throwaway container: it patches the
installed vLLM first).

    ./run_in_image.sh bash -c 'python3 patch/patch_vllm_sparse_indexer_sm120.py $VLLM && \
        VLLM_MOET_SPARSE_INDEXER=1 VLLM_MOET_SPARSE_INDEXER_DIR=$PWD python3 tests/test_vllm_patch.py'

Checks: the wrappers resolve to sm120_sparse_indexer; SparseMQAIndexer.weights_dtype is fp32; vLLM's own
sparse_mqa_logits_paged_decode / sparse_mqa_logits_prefill_chunk (expansion, metadata, logits, top-k, remap, and the
kernel_metadata reuse of later layers) write exactly the dense path's top-k indices.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CBS, NCAND, PAGE_KV, TOPK, ceil_to, dense_paged_logits, dense_prefill_logits, det_topk, make_candidates,
    make_paged_cache, quant_view, rand_q, rand_scales,
)


def main():
    assert os.environ.get("VLLM_MOET_SPARSE_INDEXER") == "1"
    import vllm.utils.deep_gemm as dg
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        sparse_mqa_logits_paged_decode, sparse_mqa_logits_prefill_chunk,
    )
    from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer

    torch.cuda.init()
    m = dg.moet_sm120_sparse_indexer()
    assert m is not None, "module did not load"
    assert dg.has_deep_gemm_sparse_mqa()
    dg._lazy_init()
    assert dg._fp8_fp4_paged_sparse_mqa_logits_impl is m.fp8_fp4_paged_sparse_mqa_logits
    assert dg._get_sparse_mqa_logits_metadata_impl is m.get_sparse_mqa_logits_metadata
    assert SparseMQAIndexer.weights_dtype == torch.float32, SparseMQAIndexer.weights_dtype
    print("ok   wrappers -> sm120_sparse_indexer, SparseMQAIndexer.weights_dtype = fp32")
    fails = 0
    gen = torch.Generator(device="cuda").manual_seed(99)

    # ---- decode: 40 rows (5 requests x 8 spec rows), contexts ~ 3000 / 70000, two consumer layers
    pool = 2048
    kv3 = make_paged_cache(pool, gen)
    kv4 = quant_view(kv3)
    for L in (3000, 70000):
        rows = 40
        ctx = torch.tensor([L - 7 + r % 8 + 13 * (r // 8) for r in range(rows)], dtype=torch.int32, device="cuda")
        npages = (int(ctx.max()) + PAGE_KV - 1) // PAGE_KV
        bt = torch.randint(0, pool, (5, npages), dtype=torch.int32, device="cuda",
                           generator=gen).repeat_interleave(8, dim=0).contiguous()
        cands = make_candidates(ctx, gen)
        zeros = torch.zeros(rows, dtype=torch.int32, device="cuda")
        si = torch.zeros(rows, NCAND, dtype=torch.int32, device="cuda")
        end = torch.zeros(rows, dtype=torch.int32, device="cuda")
        cols = torch.zeros(rows, TOPK, dtype=torch.int32, device="cuda")
        meta = None
        for layer in range(2):
            q, qs, w = rand_q(rows, gen)
            out = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
            meta = sparse_mqa_logits_paged_decode(
                q.unsqueeze(1), qs.unsqueeze(1), kv4, w, ctx, bt, torch.arange(rows, dtype=torch.int32, device="cuda"),
                cands, CBS, 8, TOPK, out, row_ks=zeros, sparse_indices=si, end=end, col_indices=cols,
                kernel_metadata=meta)
            dense = dense_paged_logits(q, qs, kv4, w, ctx, bt, ceil_to(int(ctx.max()), 256))
            apply_candidate_mask(dense, None, ctx, cands, CBS)
            ref = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
            det_topk().topk_decode(dense, ctx.view(-1, 1), 1, ref, TOPK)
            bad = int((out != ref).any(1).sum())
            fails += bad > 0
            print(f"{'ok' if not bad else 'FAIL':4s} vLLM sparse_mqa_logits_paged_decode ctx~{L} layer {layer} "
                  f"({'reused' if layer else 'new'} metadata): {rows - bad}/{rows} rows equal to dense")

    # ---- prefill chunk: two requests in the workspace, the second at an unaligned offset
    LA, LB = 20003, 9001
    T = LA + LB
    kq = torch.randint(0, 256, (T, 64), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8)
    ksc = rand_scales(T, gen)
    ks_l, ke_l = [], []
    for off, L, n in ((0, LA, 30), (LA, LB, 34)):
        for p in range(L - n, L):
            ks_l.append(off)
            ke_l.append(off + p + 1)
    cu_ks = torch.tensor(ks_l, dtype=torch.int32, device="cuda")
    cu_ke = torch.tensor(ke_l, dtype=torch.int32, device="cuda")
    rows = cu_ks.numel()
    local = cu_ke - cu_ks
    cands = make_candidates(local, gen)
    si = torch.zeros(rows, NCAND, dtype=torch.int32, device="cuda")
    end = torch.zeros(rows, dtype=torch.int32, device="cuda")
    cols = torch.zeros(rows, TOPK, dtype=torch.int32, device="cuda")
    meta = None
    for layer in range(2):
        q, qs, w = rand_q(rows, gen)
        out = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        # as SparseMQAIndexer.forward passes them: k_quant as int8, k_scale as int32 squeezed
        meta = sparse_mqa_logits_prefill_chunk(
            q, qs, kq.view(torch.int8), ksc.view(torch.uint8).view(T, 4).view(torch.int32).squeeze(-1), w, cu_ks,
            cu_ke, cands, CBS, 8, TOPK, out, sparse_indices=si, end=end, col_indices=cols, kernel_metadata=meta)
        dense = dense_prefill_logits(q, qs, kq.view(torch.int8), ksc, w, cu_ks, cu_ke, int(local.max()))
        apply_candidate_mask(dense, None, local, cands, CBS)
        ref = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        det_topk().topk_prefill(dense, None, local.contiguous(), ref, rows, TOPK, True)
        bad = int((out != ref).any(1).sum())
        fails += bad > 0
        print(f"{'ok' if not bad else 'FAIL':4s} vLLM sparse_mqa_logits_prefill_chunk layer {layer}: "
              f"{rows - bad}/{rows} rows equal to dense")
    print("ALL EXACT" if not fails else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
