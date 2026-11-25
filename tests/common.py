"""Inputs shaped like DeepSeek-V4.1-Flash's indexer on sm_120 (MXFP4 cache, 32 heads x 128 dims, 128-row pages).

Runs inside the vllm-moet image (it imports vLLM's DeepGEMM wrappers and the indexer's Triton pieces).
"""

from __future__ import annotations

import math
import os
import sys

import torch

H, D, ROWB = 32, 128, 64          # index_n_heads, index_head_dim, packed bytes per row
HEAD_BYTES = ROWB + 4             # + 4 UE8M0 scale bytes
PAGE_KV = 128                     # ratio-1 indexer layers' kernel block
TOPK = 512                        # index_topk
CBS = 8                           # candidate_block_size
NCAND = 2048                      # candidate_topk_blocks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sm120_sparse_indexer as sp  # noqa: E402


def det_topk():
    return sp._det_topk()


def rand_scales(n: int, gen: torch.Generator, lo: int = 122, hi: int = 132) -> torch.Tensor:
    """n packed int32 UE8M0 quadruples (2^(e-127), e in [lo, hi])."""
    b = torch.randint(lo, hi + 1, (n, 4), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8)
    return b.view(torch.int32).view(n)


def rand_q(rows: int, gen: torch.Generator):
    """(values int8 [rows, 32, 64], scales int32 [rows, 32], weights fp32 [rows, 32])."""
    q = torch.randint(0, 256, (rows, H, ROWB), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8)
    qs = rand_scales(rows * H, gen).view(rows, H)
    w = torch.randn(rows, H, device="cuda", generator=gen) * 0.05
    return q.view(torch.int8), qs, w


def make_paged_cache(num_pages: int, gen: torch.Generator) -> torch.Tensor:
    """vLLM's indexer cache allocation [num_pages, 128, 68] uint8: per page 128 x 64 B of e2m1 pairs, then 128 x 4
    UE8M0 bytes (DeepGEMM's paged layout)."""
    kv = torch.empty(num_pages, PAGE_KV, HEAD_BYTES, dtype=torch.uint8, device="cuda")
    flat = kv.view(num_pages, PAGE_KV * HEAD_BYTES)
    flat[:, : PAGE_KV * ROWB] = torch.randint(0, 256, (num_pages, PAGE_KV * ROWB), dtype=torch.int32, device="cuda",
                                              generator=gen).to(torch.uint8)
    flat[:, PAGE_KV * ROWB:] = rand_scales(num_pages * PAGE_KV, gen).view(torch.uint8).view(num_pages, PAGE_KV * 4)
    return kv


def quant_view(kv3: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view

    return kv_cache_as_quant_view(kv3, D, use_fp4_cache=True)


def make_candidates(ctx: torch.Tensor, gen: torch.Generator, ncand: int = NCAND, cbs: int = CBS) -> torch.Tensor:
    """Per row: all blocks of [0, ctx) when they number <= ncand, else the newest block plus ncand - 1 distinct random
    others; ascending, -1 padded (the shape select_candidate_blocks' deterministic top-k writes)."""
    rows = ctx.numel()
    out = torch.full((rows, ncand), -1, dtype=torch.int32, device="cuda")
    for r, L in enumerate(ctx.tolist()):
        nb = (L + cbs - 1) // cbs
        if nb <= 0:
            continue
        if nb <= ncand:
            out[r, :nb] = torch.arange(nb, dtype=torch.int32, device="cuda")
        else:
            others = torch.randperm(nb - 1, device="cuda", generator=gen)[: ncand - 1]
            sel = torch.cat([others, torch.tensor([nb - 1], device="cuda")]).sort().values
            out[r] = sel.to(torch.int32)
    return out


def ceil_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def dense_schedule(ctx) -> torch.Tensor:
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    return get_paged_mqa_logits_metadata(ctx.view(-1, 1), PAGE_KV, num_sms())


def dense_paged_logits(q, qs, kv4, w, ctx, bt, width: int, sched=None) -> torch.Tensor:
    """DeepGEMM's sm_120 dense paged MQA logits as vLLM's decode calls them (flattened rows, next_n = 1); the
    schedule is built once per step by the metadata builder in vLLM."""
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

    ctx2 = ctx.view(-1, 1)
    if sched is None:
        sched = dense_schedule(ctx)
    return fp8_fp4_paged_mqa_logits((q.unsqueeze(1), qs.unsqueeze(1)), kv4, w, ctx2, bt, sched, max_model_len=width)


def dense_prefill_logits(q, qs, kq, ksc, w, cu_ks, cu_ke, max_seqlen_k: int) -> torch.Tensor:
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits((q, qs), (kq, ksc), w, cu_ks, cu_ke, max_seqlen_k)


def remap(cols, si, row_ks, sbk: int, topk: int, width: int, out):
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import _sparse_topk_remap_kernel
    from vllm.triton_utils import triton

    rows = cols.shape[0]
    _sparse_topk_remap_kernel[(rows,)](cols, cols.stride(0), si, si.stride(0), row_ks, row_ks.stride(0), out,
                                       out.stride(0), topk, width, SBK=sbk, K_POW2=triton.next_power_of_2(topk))
    return out


def expand(cands, row_ks, row_ke, cbs: int, sbk: int):
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import candidate_blocks_to_sparse_indices

    return candidate_blocks_to_sparse_indices(cands, row_ks, row_ke, cbs, sbk)


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


def cuda_time(fn, iters: int = 20, warmup: int = 3) -> float:
    """Median us per call (events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]
