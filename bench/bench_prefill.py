#!/usr/bin/env python3
"""Prefill-chunk cost of the indexer layers on sm_120: today's dense path vs block_topk, eager, as the engine chunks.

One prefill step = the last --rows query tokens of one request of --ctx tokens (row i sees the K of positions
[0, ctx - rows + i]; a ratio-2 layer half as many keys). The engine splits the step's rows into chunks of at most
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (512) / (4 B x the request's keys) rows (`_split_indexer_prefill_chunks`) and runs,
per chunk and layer:
  dense  fp8_fp4_mqa_logits fp32 [chunk rows, max_seqlen_k] -> (select_candidate_blocks | apply_candidate_mask) ->
         det_topk prefill
  block  block_topk.contiguous_topk (ratio-2 layers, the candidate source) / contiguous_candidates_topk (consumers)
The K gather into the packed workspace is the same for both and not timed. Times are wall-clock milliseconds per step
(torch.cuda.synchronize around the chunk loop: eager launches and host overhead included), best of --iters; "host" is
the part until the last launch returned (CPU side only: when it nears the total, the path is launch-bound).
A ratio-2 layer's rows are sharded over the TP ranks from indexer_tp_shard_min_rows (2048) rows: --plain-shard 4 times
one rank's quarter of every chunk. "step" = 3 ratio-2 layers + the source + 4 consumers.

Every chunk's top-k rows (and the source's candidate blocks) are compared between the two paths.

    python3 bench/bench_prefill.py [--ctx 16384,131072,400000,700000] [--rows 8192] [--plain-shard 4] [--iters 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from common import CBS, NCAND, TOPK, det_topk, rand_q, rand_scales  # noqa: E402
from sm120_sparse_indexer import block_topk as bk  # noqa: E402

MAX_LOGITS_ELEMS = 512 * 1024 * 1024 // 4


def wall_ms(fn, iters: int) -> tuple[float, float]:
    """(total, host) milliseconds of the best of `iters` runs: host = until the last launch returned (the eager
    path's CPU side: Python, argument checks, driver calls), total = until the GPU finished."""
    fn()
    torch.cuda.synchronize()
    best = (float("inf"), 0.0)
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        best = min(best, ((t2 - t0) * 1e3, (t1 - t0) * 1e3))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="16384,131072,400000,700000", help="request length in tokens")
    ap.add_argument("--rows", type=int, default=8192, help="query tokens of the step")
    ap.add_argument("--plain-shard", type=int, default=4, help="TP ranks sharing a ratio-2 layer's rows (1: none)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--kinds", default="plain,source,consumer")
    args = ap.parse_args()
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask, select_candidate_blocks
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()}, free {torch.cuda.mem_get_info()[0] >> 20} MiB, block_topk kernel info "
          f"[scan regs, scan CTAs/SM, rescore regs, rescore CTAs/SM] {bk._ext().kernel_info(0)}, step rows {args.rows}, "
          f"plain shard 1/{args.plain_shard}", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(5)
    dt = det_topk()
    i32 = dict(dtype=torch.int32, device="cuda")
    print(f"{'ctx':>8} {'kind':>8} {'keys':>7} {'chunks x rows':>14} | {'dense ms':>9} {'(host)':>7} {'(logits)':>9} | "
          f"{'block ms':>9} {'(host)':>7} {'(scan)':>8} | {'dense/block':>11} | equal", flush=True)
    for ctx in [int(x) for x in args.ctx.split(",")]:
        rows = min(args.rows, ctx)
        q, qs, w = rand_q(rows, gen)
        step = {}
        cand_src = None
        for kind in args.kinds.split(","):
            ratio = 2 if kind == "plain" else 1
            N = ctx // ratio
            kq = torch.randint(0, 256, (N, 64), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8).view(torch.int8)
            ksc = rand_scales(N, gen)
            cu_ks = torch.zeros(rows, **i32)
            cu_ke = ((torch.arange(ctx - rows, ctx, device="cuda") + 1) // ratio).to(torch.int32)
            row_ends = cu_ke  # ks = 0
            chunk_rows = max(1, MAX_LOGITS_ELEMS // N)
            chunks = [(r0, min(rows, r0 + chunk_rows)) for r0 in range(0, rows, chunk_rows)]

            def width_of(b: int) -> int:  # the chunk's max_seqlen_k, a host number in the engine (no device read)
                return (ctx - rows + b) // ratio

            if kind == "plain" and args.plain_shard > 1 and rows >= 2048:
                # one TP rank's contiguous quarter of the step's rows, narrowed per chunk
                lo, hi = 0, rows // args.plain_shard
                chunks = [(max(a, lo), min(b, hi)) for a, b in chunks if max(a, lo) < min(b, hi)]
            out_d = torch.full((rows, TOPK), -1, **i32)
            out_b = torch.full((rows, TOPK), -1, **i32)
            cand_d = torch.full((rows, NCAND), -1, **i32) if kind == "source" else None
            cand_b = torch.full((rows, NCAND), -1, **i32) if kind == "source" else None
            if kind == "consumer":
                if cand_src is None:  # the source's candidates for these rows (another q), by the block path
                    q2, qs2, w2 = rand_q(rows, gen)
                    cand_src = torch.full((rows, NCAND), -1, **i32)
                    for a, b in chunks:
                        bk.contiguous_topk(q2[a:b], qs2[a:b], kq, ksc, w2[a:b], cu_ks[a:b], cu_ke[a:b],
                                           width_of(b), TOPK, out_b[a:b], candidates=cand_src[a:b])

            def dense(logits_only=False):
                for a, b in chunks:
                    ends = row_ends[a:b]
                    lg = fp8_fp4_mqa_logits((q[a:b], qs[a:b]), (kq, ksc), w[a:b], cu_ks[a:b], cu_ke[a:b],
                                            width_of(b))
                    if logits_only:
                        continue
                    if kind == "source":
                        select_candidate_blocks(lg, None, ends, NCAND, CBS, cand_d[a:b])
                    elif kind == "consumer":
                        apply_candidate_mask(lg, None, ends, cand_src[a:b], CBS)
                    dt.topk_prefill(lg, None, ends, out_d[a:b], b - a, TOPK, True)
                    del lg

            def block():
                for a, b in chunks:
                    width = width_of(b)
                    if kind == "consumer":
                        bk.contiguous_candidates_topk(q[a:b], qs[a:b], kq, ksc, w[a:b], cu_ks[a:b], cu_ke[a:b], width,
                                                      TOPK, out_b[a:b], cand_src[a:b])
                    else:
                        bk.contiguous_topk(q[a:b], qs[a:b], kq, ksc, w[a:b], cu_ks[a:b], cu_ke[a:b], width, TOPK,
                                           out_b[a:b], candidates=None if cand_b is None else cand_b[a:b])

            def scan_only():
                bk._STOP_AFTER = 1
                block()
                bk._STOP_AFTER = 0

            t_dense, h_dense = wall_ms(dense, args.iters)
            t_dlog = wall_ms(lambda: dense(True), args.iters)[0]
            t_block, h_block = wall_ms(block, args.iters)
            t_scan = wall_ms(scan_only, args.iters)[0] if kind != "consumer" else float("nan")
            block()
            dense()
            torch.cuda.synchronize()
            cover = torch.zeros(rows, dtype=torch.bool, device="cuda")
            for a, b in chunks:
                cover[a:b] = True
            equal = bool((out_d[cover] == out_b[cover]).all()) and (cand_d is None or bool((cand_d[cover] == cand_b[cover]).all()))
            n_rows = sum(b - a for a, b in chunks)
            print(f"{ctx:>8} {kind:>8} {N:>7} {len(chunks):>4} x {chunks[0][1] - chunks[0][0]:>5}{'' if n_rows == rows else '*':1s}  | "
                  f"{t_dense:>9.2f} {h_dense:>7.2f} {t_dlog:>9.2f} | {t_block:>9.2f} {h_block:>7.2f} {t_scan:>8.2f} | "
                  f"{t_dense / t_block:>10.2f}x | {'yes' if equal else 'NO'}", flush=True)
            step[kind] = (t_dense, t_block)
            del kq, ksc, out_d, out_b, cand_d, cand_b
            torch.cuda.empty_cache()
        if all(k in step for k in ("plain", "source", "consumer")):
            d = 3 * step["plain"][0] + step["source"][0] + 4 * step["consumer"][0]
            b = 3 * step["plain"][1] + step["source"][1] + 4 * step["consumer"][1]
            print(f"{ctx:>8} {'step':>8} {'':>7} {'8 layers':>14} | {d:>9.2f} {'':>7} {'':>9} | {b:>9.2f} {'':>7} {'':>8} | "
                  f"{d / b:>10.2f}x |", flush=True)
        del q, qs, w, cand_src
        torch.cuda.empty_cache()
    print(f"peak allocated {torch.cuda.max_memory_allocated() >> 20} MiB")


if __name__ == "__main__":
    main()
