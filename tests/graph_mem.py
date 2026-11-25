#!/usr/bin/env python3
"""CUDA-graph memory of one decode step's indexer layers, single stream, as vLLM's caching allocator lays it out.

Emulates the eight indexer calls of DeepSeek-V4.1-Flash's decode step in capture order with the engine's shapes:
  layers 2, 8, 14   ratio 2 (64-row pages), width = max_model_len / 2
  layer 20          ratio 1 (128-row pages), the candidate source, width = max_model_len
  layers 24..36     ratio 1, candidate consumers
in three modes:
  dense    today: fp32 logits [rows, width] -> (select_candidate_blocks | apply_candidate_mask) -> det_topk
  sparse   the consumers on the sparse MQA-logits API (indexer_sparse_logits), the other four layers dense
  block    all eight on block_topk (VLLM_MOET_INDEXER_BLOCK_TOPK=1): block maxima + the selected blocks' logits
and reports the graph pool's reserved bytes per row (slope over two row counts). Run with the engine's allocator
(ALLOC_CONF= ./run_in_image.sh ...: PyTorch's default, no expandable segments).

    ALLOC_CONF= ./run_in_image.sh python3 tests/graph_mem.py [--max-model-len 1048576] [--rows 16,48]
        [--block-rows 256,512] [--modes dense,sparse,block]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CBS, NCAND, TOPK, det_topk, num_sms, quant_view, rand_q, sp  # noqa: E402
from sm120_sparse_indexer import block_topk as bk  # noqa: E402
from test_block_topk import make_cache  # noqa: E402


def dense_logits(q4, qs4, kv4, w, ctx, bt, width, sched):
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

    return fp8_fp4_paged_mqa_logits((q4, qs4), kv4, w, ctx.view(-1, 1), bt, sched, max_model_len=width)


def step(mode, rows, mml, d):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask, select_candidate_blocks
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        _sparse_topk_remap_kernel, candidate_blocks_to_sparse_indices,
    )

    dt = det_topk()
    q4, qs4, w, out, cands = d["q4"], d["qs4"], d["w"], d["topk"], d["cands"]
    for _ in range(3):  # layers 2, 8, 14
        if mode == "block":
            bk.paged_topk(q4, qs4, d["kv2"], w, d["ctx2"], d["bt2"], mml // 2, TOPK, out)
        else:
            lg = dense_logits(q4, qs4, d["kv2"], w, d["ctx2"], d["bt2"], mml // 2, d["sched2"])
            dt.topk_decode(lg, d["ctx2"].view(-1, 1), 1, out, TOPK)
            del lg
    # layer 20
    if mode == "block":
        bk.paged_topk(q4, qs4, d["kv1"], w, d["ctx1"], d["bt1"], mml, TOPK, out, candidates=cands)
    else:
        lg = dense_logits(q4, qs4, d["kv1"], w, d["ctx1"], d["bt1"], mml, d["sched1"])
        select_candidate_blocks(lg, None, d["ctx1"], NCAND, CBS, cands)
        dt.topk_decode(lg, d["ctx1"].view(-1, 1), 1, out, TOPK)
        del lg
    meta = None
    for _ in range(4):  # layers 24, 28, 32, 36
        if mode == "block":
            bk.paged_candidates_topk(q4, qs4, d["kv1"], w, d["ctx1"], d["bt1"], mml, TOPK, out, cands)
        elif mode == "sparse":
            si, end, cols, zeros = d["si"], d["end"], d["cols"], d["zeros"]
            if meta is None:
                candidate_blocks_to_sparse_indices(cands, zeros, d["ctx1"], CBS, 8, out=(si, end))
                meta = sp.get_paged_sparse_mqa_logits_metadata(d["ctx1"], d["bt1"], None, 128, si, torch.int8, 8)
            lg = sp.fp8_fp4_paged_sparse_mqa_logits((q4, qs4), d["kv1"], w, meta, NCAND, 8)
            sp.sparse_topk(lg, end, TOPK, cols)
            _sparse_topk_remap_kernel[(rows,)](cols, cols.stride(0), si, si.stride(0), zeros, 1, out, out.stride(0),
                                               TOPK, lg.shape[1], SBK=8, K_POW2=512)
            del lg
        else:
            lg = dense_logits(q4, qs4, d["kv1"], w, d["ctx1"], d["bt1"], mml, d["sched1"])
            apply_candidate_mask(lg, None, d["ctx1"], cands, CBS)
            dt.topk_decode(lg, d["ctx1"].view(-1, 1), 1, out, TOPK)
            del lg


def measure(mode, rows, mml, kv1, kv2, gen):
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    L = 20000
    i32 = dict(dtype=torch.int32, device="cuda")
    ctx1 = torch.full((rows,), L, **i32)
    ctx2 = torch.full((rows,), L // 2, **i32)
    bt1 = torch.randint(0, kv1.shape[0], (rows, (L + 127) // 128), generator=gen, **i32)
    bt2 = torch.randint(0, kv2.shape[0], (rows, (L // 2 + 63) // 64), generator=gen, **i32)
    q, qs, w = rand_q(rows, gen)
    d = dict(q4=q.unsqueeze(1), qs4=qs.unsqueeze(1), w=w, kv1=kv1, kv2=kv2, ctx1=ctx1, ctx2=ctx2, bt1=bt1, bt2=bt2,
             sched1=get_paged_mqa_logits_metadata(ctx1.view(-1, 1), 128, num_sms()),
             sched2=get_paged_mqa_logits_metadata(ctx2.view(-1, 1), 64, num_sms()),
             topk=torch.empty(rows, TOPK, **i32), cands=torch.full((rows, NCAND), -1, **i32),
             si=torch.empty(rows, NCAND, **i32), end=torch.empty(rows, **i32), cols=torch.empty(rows, TOPK, **i32),
             zeros=torch.zeros(rows, **i32))
    step(mode, rows, mml, d)  # warm-up (JIT, det_topk cluster info)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            step(mode, rows, mml, d)
    torch.cuda.synchronize()
    pool = torch.cuda.memory_reserved() - before
    g.replay()
    torch.cuda.synchronize()
    del g
    torch.cuda.empty_cache()
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-len", type=int, default=1 << 20)
    ap.add_argument("--rows", default="16,48", help="two row counts for the dense and sparse modes (6 MiB per row at 1M)")
    ap.add_argument("--block-rows", default="256,512", help="two row counts for the block mode")
    ap.add_argument("--modes", default="dense,sparse,block")
    args = ap.parse_args()
    torch.cuda.init()
    gen = torch.Generator(device="cuda").manual_seed(3)
    kv1 = quant_view(make_cache(512, 128, gen))
    kv2 = quant_view(make_cache(1024, 64, gen))
    mml = args.max_model_len
    # the engine's registration at layer construction (SparseAttnIndexer.__init__): one scratch size per decode call
    bk.reserve_paged(mml // 2, TOPK, TOPK, False)
    bk.reserve_paged(mml, NCAND, TOPK, True)
    print(f"allocator: PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}, max_model_len {mml}, "
          f"block scratch {bk._PAGED_ROW_WORDS * 4 / 2**20:.3f} MiB per row per call", flush=True)
    scale = (1 << 20) / mml
    for mode in args.modes.split(","):
        r0, r1 = [int(x) for x in (args.block_rows if mode == "block" else args.rows).split(",")]
        m0 = measure(mode, r0, mml, kv1, kv2, gen)
        m1 = measure(mode, r1, mml, kv1, kv2, gen)
        per_row = (m1 - m0) / (r1 - r0) / 2**20
        print(f"{mode:6s}: graph pool {m0 / 2**20:.1f} MiB at {r0} rows, {m1 / 2**20:.1f} MiB at {r1} rows -> "
              f"{per_row:.3f} MiB/row" + (f" = {per_row * scale:.2f} MiB/row at 1,048,576" if scale != 1 else ""),
              flush=True)


if __name__ == "__main__":
    main()
