#!/usr/bin/env python3
"""Decode indexer microbenchmark on sm_120: sparse (this module) vs dense (today's path), per consumer-layer call.

  dense         DeepGEMM fp8_fp4_paged_mqa_logits [rows, W] fp32 -> apply_candidate_mask -> det_topk decode
  sparse/first  expand candidates -> metadata -> sparse logits [rows, 16384] fp32 -> det_topk -> remap
                (the first consumer layer of a step)
  sparse/reuse  sparse logits -> det_topk -> remap (layers 2..4 of the step reuse the expansion and the metadata)

Rows are flattened spec-decode rows (8 per request, contexts L - 7 + i); every request's pages are drawn at random
from one shared pool (default 46 MiB: L2-resident, so both paths' KV reads are optimistic; the memory budget of a
side job on a serving GPU does not allow a pool larger than the 128 MB L2).
Times are CUDA-graph replays (median of N); the dense path runs in row chunks when its logits would not fit the memory
budget and the chunk times are summed. Memory: the logits bytes each path allocates per call, and for the dense path
the CUDA-graph figure, which is sized by max_model_len (1,048,576), not by the context.

    python3 bench/bench.py [--rows 8,128,384,512] [--ctx 65536,262144,1048576] [--pool-pages 24576] [--iters 20]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from common import (  # noqa: E402
    CBS, NCAND, PAGE_KV, TOPK, ceil_to, dense_paged_logits, dense_schedule, det_topk, make_candidates, make_paged_cache, quant_view,
    rand_q, sp,
)

MAX_MODEL_LEN = 1 << 20
DENSE_CHUNK_BYTES = 24 << 20


def graph_time(fn, iters: int) -> float:
    """Median microseconds per replay of fn captured in a CUDA graph (eager median if capture fails)."""
    fn()
    torch.cuda.synchronize()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    except Exception as exc:  # noqa: BLE001
        print(f"  (graph capture failed: {exc!r}; eager timing)")
        run = fn
    run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    del run
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,128,384,512")
    ap.add_argument("--ctx", default="65536,262144,1048576")
    ap.add_argument("--pool-pages", type=int, default=5504, help="KV pool pages (5504 x 8704 B = 46 MiB, L2-resident)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        _sparse_topk_remap_kernel, candidate_blocks_to_sparse_indices,
    )
    from vllm.triton_utils import triton

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()}, free {torch.cuda.mem_get_info()[0] >> 20} MiB, L2 "
          f"{torch.cuda.get_device_properties(0).L2_cache_size >> 20} MiB", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(7)
    kv3 = make_paged_cache(args.pool_pages, gen)
    kv4 = quant_view(kv3)
    dt = det_topk()
    S = NCAND
    print(f"{'ctx':>8} {'rows':>5} | {'dense us':>9} {'(logits)':>9} | {'sparse 1st':>10} {'reuse':>7} {'(logits)':>9} | "
          f"{'speedup':>7} | {'dense graph MiB':>15} {'sparse MiB':>10}", flush=True)
    for L in [int(x) for x in args.ctx.split(",")]:
        for rows in [int(x) for x in args.rows.split(",")]:
            nreq = (rows + 7) // 8
            ctx = torch.tensor([L - 7 + r % 8 for r in range(rows)], dtype=torch.int32, device="cuda")
            npages = (L + PAGE_KV) // PAGE_KV
            bt = torch.randint(0, args.pool_pages, (nreq, npages), dtype=torch.int32, device="cuda",
                               generator=gen).repeat_interleave(8, dim=0)[:rows].contiguous()
            q, qs, w = rand_q(rows, gen)
            q4, qs4 = q.unsqueeze(1), qs.unsqueeze(1)
            cands = make_candidates(ctx, gen)
            zeros = torch.zeros(rows, dtype=torch.int32, device="cuda")
            si = torch.empty(rows, S, dtype=torch.int32, device="cuda")
            end = torch.empty(rows, dtype=torch.int32, device="cuda")
            cols = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            out = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            meta_box = [sp.get_paged_sparse_mqa_logits_metadata(ctx, bt, None, PAGE_KV, si, torch.int8, 8)]

            def topk_remap(logits):
                sp.sparse_topk(logits, end, TOPK, cols)
                _sparse_topk_remap_kernel[(rows,)](cols, cols.stride(0), si, si.stride(0), zeros, 1, out,
                                                   out.stride(0), TOPK, logits.shape[1], SBK=8,
                                                   K_POW2=triton.next_power_of_2(TOPK))

            def sparse_first():
                candidate_blocks_to_sparse_indices(cands, zeros, ctx, CBS, 8, out=(si, end))
                meta = sp.get_paged_sparse_mqa_logits_metadata(ctx, bt, None, PAGE_KV, si, torch.int8, 8)
                topk_remap(sp.fp8_fp4_paged_sparse_mqa_logits((q4, qs4), kv4, w, meta, S, 8))

            def sparse_reuse():
                topk_remap(sp.fp8_fp4_paged_sparse_mqa_logits((q4, qs4), kv4, w, meta_box[0], S, 8))

            def sparse_logits():
                sp.fp8_fp4_paged_sparse_mqa_logits((q4, qs4), kv4, w, meta_box[0], S, 8)

            candidate_blocks_to_sparse_indices(cands, zeros, ctx, CBS, 8, out=(si, end))
            meta_box[0] = sp.get_paged_sparse_mqa_logits_metadata(ctx, bt, None, PAGE_KV, si, torch.int8, 8)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            sparse_first()
            torch.cuda.synchronize()
            sparse_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
            t_first = graph_time(sparse_first, args.iters)
            t_reuse = graph_time(sparse_reuse, args.iters)
            t_slog = graph_time(sparse_logits, args.iters)

            # dense, in row chunks
            t_dense = t_dlog = float("nan")
            if not args.no_dense:
                width = ceil_to(L, 256)
                chunk = max(1, min(rows, DENSE_CHUNK_BYTES // (width * 4)))
                t_dense = t_dlog = 0.0
                seen = {}  # identical chunk shapes are timed once
                for r0 in range(0, rows, chunk):
                    r1 = min(rows, r0 + chunk)
                    if r1 - r0 in seen:
                        t_dense += seen[r1 - r0][0]
                        t_dlog += seen[r1 - r0][1]
                        continue
                    sl = slice(r0, r1)
                    out_d = torch.empty(r1 - r0, TOPK, dtype=torch.int32, device="cuda")
                    sched = dense_schedule(ctx[sl])

                    def dense_full():
                        lg = dense_paged_logits(q[sl], qs[sl], kv4, w[sl], ctx[sl], bt[sl], width, sched)
                        apply_candidate_mask(lg, None, ctx[sl], cands[sl], CBS)
                        dt.topk_decode(lg, ctx[sl].view(-1, 1), 1, out_d, TOPK)

                    def dense_logits():
                        dense_paged_logits(q[sl], qs[sl], kv4, w[sl], ctx[sl], bt[sl], width, sched)

                    seen[r1 - r0] = (graph_time(dense_full, args.iters), graph_time(dense_logits, args.iters))
                    t_dense += seen[r1 - r0][0]
                    t_dlog += seen[r1 - r0][1]
                    torch.cuda.empty_cache()
            # CUDA-graph memory of the dense path at max_model_len: logits + candidate flags
            dense_graph_mib = rows * (MAX_MODEL_LEN * 4 + (MAX_MODEL_LEN // CBS + 1)) / 2**20
            print(f"{L:>8} {rows:>5} | {t_dense:>9.1f} {t_dlog:>9.1f} | {t_first:>10.1f} {t_reuse:>7.1f} {t_slog:>9.1f} | "
                  f"{t_dense / t_reuse:>6.1f}x | {dense_graph_mib:>15.0f} {sparse_mib:>10.1f}", flush=True)
            del q, qs, w, cands, bt, si, end, cols, out, meta_box
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
