#!/usr/bin/env python3
"""Per-call time of the dense indexer layers on sm_120: today's dense path vs block_topk (this module).

  dense  DeepGEMM fp8_fp4_paged_mqa_logits fp32 [rows, W] (+ select_candidate_blocks for the candidate source,
         apply_candidate_mask for a consumer) -> det_topk decode   (W = the context rounded up; the engine's W is
         max_model_len, which costs the dense path more: its candidate kernels run over W)
  block  source / ratio2: scan -> det_topk over the block maxima -> rescore -> det_topk -> finish
         consumer:        compact -> rescore -> det_topk -> finish
         (width 1,048,576 for the ratio-1 layers, 524,288 for a ratio-2 layer: the engine's static shapes)

Rows are flattened spec-decode rows (8 per request, contexts L - 7 + i). Every request's pages are drawn at random from
one shared pool: --pool-mib 46 (default) is L2-resident (the kernels' compute rate); a pool larger than the 128 MiB L2
makes the reads DRAM-bound when the requests' pages are distinct (--distinct: consecutive pages per request).
Times are CUDA-graph replays (median); the dense path runs in row chunks of at most --dense-chunk-mib of logits and
the chunk times are summed. GB/s = rows * context * 68 B / time (the K bytes a call scores per second).

    python3 bench/bench_block_topk.py [--kind source,ratio2] [--rows 8,128,384,512] [--ctx 65536,262144,1048576]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from common import (  # noqa: E402
    CBS, HEAD_BYTES, NCAND, ROWB, TOPK, ceil_to, det_topk, make_candidates, num_sms, quant_view, rand_q, rand_scales,
)
from sm120_sparse_indexer import block_topk as bk  # noqa: E402


def graph_time(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    del g
    return ts[len(ts) // 2]


def make_cache(num_pages: int, page_kv: int, gen) -> torch.Tensor:
    kv = torch.empty(num_pages, page_kv, HEAD_BYTES, dtype=torch.uint8, device="cuda")
    flat = kv.view(num_pages, page_kv * HEAD_BYTES)
    step = 4096
    for p0 in range(0, num_pages, step):
        n = min(step, num_pages - p0)
        flat[p0:p0 + n, : page_kv * ROWB] = torch.randint(0, 256, (n, page_kv * ROWB), dtype=torch.int32, device="cuda",
                                                         generator=gen).to(torch.uint8)
        flat[p0:p0 + n, page_kv * ROWB:] = rand_scales(n * page_kv, gen).view(torch.uint8).view(n, page_kv * 4)
    return kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="source,ratio2,consumer")
    ap.add_argument("--rows", default="8,128,384,512")
    ap.add_argument("--ctx", default="65536,262144,1048576", help="context in tokens (a ratio-2 layer scores half)")
    ap.add_argument("--pool-mib", type=float, default=46.0)
    ap.add_argument("--distinct", action="store_true", help="consecutive pool pages per request (no page reuse)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dense-chunk-mib", type=int, default=192)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--launch", default="",
                    help="scan_threads,scan_ctas_per_sm,rescore_threads,rescore_ctas_per_sm,scan_chunk")
    ap.add_argument("--group", type=int, default=8, help="rows per request (block_topk.set_decode_group)")
    ap.add_argument("--parts", action="store_true", help="also time the scan alone")
    ap.add_argument("--steps", action="store_true", help="time the block path's steps (differences of prefixes)")
    args = ap.parse_args()
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask, select_candidate_blocks
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata

    torch.cuda.init()
    if args.launch:
        bk._ext().set_launch(*[int(x) for x in args.launch.split(",")])
    bk.set_decode_group(args.group)
    print(f"device {torch.cuda.get_device_name()}, free {torch.cuda.mem_get_info()[0] >> 20} MiB, L2 "
          f"{torch.cuda.get_device_properties(0).L2_cache_size >> 20} MiB, pool {args.pool_mib} MiB"
          f"{' distinct pages' if args.distinct else ''}, launch {args.launch or 'default'}, group {args.group}", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(7)
    dt = det_topk()
    print(f"{'kind':>8} {'keys':>8} {'rows':>5} | {'dense us':>9} {'(logits)':>9} {'GB/s':>6} | {'block us':>9} {'(scan)':>8} "
          f"{'GB/s':>6} | {'dense/block':>11}", flush=True)
    for kind in args.kind.split(","):
        source, consumer = kind == "source", kind == "consumer"
        page_kv = 64 if kind == "ratio2" else 128
        ratio = 2 if kind == "ratio2" else 1
        pool_pages = int(args.pool_mib * 2**20 / (page_kv * HEAD_BYTES))
        kv4 = quant_view(make_cache(pool_pages, page_kv, gen))
        width_engine = (1 << 20) // ratio
        if not consumer:
            bk.reserve_paged(width_engine, NCAND if source else TOPK, TOPK, source)
        for ctx_tokens in [int(x) for x in args.ctx.split(",")]:
            L = ctx_tokens // ratio
            for rows in [int(x) for x in args.rows.split(",")]:
                nreq = (rows + 7) // 8
                ctx = torch.tensor([L - 7 + r % 8 for r in range(rows)], dtype=torch.int32, device="cuda")
                npages = (L + page_kv) // page_kv
                if args.distinct:
                    base = (torch.arange(nreq, device="cuda") * npages).view(-1, 1) + torch.arange(npages, device="cuda")
                    bt_req = (base % pool_pages).to(torch.int32)
                else:
                    bt_req = torch.randint(0, pool_pages, (nreq, npages), dtype=torch.int32, device="cuda", generator=gen)
                bt = bt_req.repeat_interleave(8, dim=0)[:rows].contiguous()
                q, qs, w = rand_q(rows, gen)
                q4, qs4 = q.unsqueeze(1), qs.unsqueeze(1)
                out = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
                cand = torch.empty(rows, NCAND, dtype=torch.int32, device="cuda") if source else None
                cand_in = make_candidates(ctx, gen) if consumer else None

                def block():
                    if consumer:
                        bk.paged_candidates_topk(q4, qs4, kv4, w, ctx, bt, width_engine, TOPK, out, cand_in)
                    else:
                        bk.paged_topk(q4, qs4, kv4, w, ctx, bt, width_engine, TOPK, out, candidates=cand)

                t_block = graph_time(block, args.iters)
                if args.steps and not consumer:
                    cum = []
                    for stop in (1, 2, 3, 4):
                        bk._STOP_AFTER = stop
                        cum.append(graph_time(block, args.iters))
                    bk._STOP_AFTER = 0
                    cum.append(t_block)
                    d = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, 5)]
                    print(f"        steps us: scan {d[0]:.1f}  block top-k {d[1]:.1f}  rescore {d[2]:.1f}  column top-k "
                          f"{d[3]:.1f}  finish {d[4]:.1f}", flush=True)
                t_scan = float("nan")
                if args.parts and not consumer:
                    nb = NCAND if source else TOPK
                    sb = bk._layout(width_engine, nb, TOPK, source)[0]
                    sc = torch.empty(rows, width_engine // 8, device="cuda")
                    lg = torch.empty(rows, sb * 8, device="cuda")
                    vb = torch.empty(rows, dtype=torch.int32, device="cuda")
                    se = torch.empty(rows, dtype=torch.int32, device="cuda")
                    en = torch.empty(rows, dtype=torch.int32, device="cuda")

                    def scan():
                        bk._ext().paged_scan(q4, qs4, kv4, w, ctx, bt, sc, vb, se, en, lg, nb, sb, width_engine, source,
                                             source, bk._DECODE_GROUP)

                    t_scan = graph_time(scan, args.iters)
                    del sc, lg
                t_dense = t_dlog = float("nan")
                if not args.no_dense:
                    width = ceil_to(L, 256)
                    chunk = max(1, min(rows, (args.dense_chunk_mib << 20) // (width * 4)))
                    t_dense = t_dlog = 0.0
                    seen = {}
                    for r0 in range(0, rows, chunk):
                        r1 = min(rows, r0 + chunk)
                        if r1 - r0 not in seen:
                            sl = slice(r0, r1)
                            out_d = torch.empty(r1 - r0, TOPK, dtype=torch.int32, device="cuda")
                            cand_d = torch.empty(r1 - r0, NCAND, dtype=torch.int32, device="cuda")
                            c2 = ctx[sl].view(-1, 1)
                            sched = get_paged_mqa_logits_metadata(c2, page_kv, num_sms())

                            def dense_logits():
                                return fp8_fp4_paged_mqa_logits((q4[sl], qs4[sl]), kv4, w[sl], c2, bt[sl], sched,
                                                                max_model_len=width)

                            def dense_full():
                                lg = dense_logits()
                                if source:
                                    select_candidate_blocks(lg, None, ctx[sl], NCAND, CBS, cand_d)
                                elif consumer:
                                    apply_candidate_mask(lg, None, ctx[sl], cand_in[sl], CBS)
                                dt.topk_decode(lg, c2, 1, out_d, TOPK)

                            seen[r1 - r0] = (graph_time(dense_full, args.iters), graph_time(dense_logits, args.iters))
                            torch.cuda.empty_cache()
                        t_dense += seen[r1 - r0][0]
                        t_dlog += seen[r1 - r0][1]
                gb = rows * L * 68 / 1e9
                gbs = float("nan") if consumer else gb / ((t_scan if args.parts else t_block) * 1e-6)
                print(f"{kind:>8} {L:>8} {rows:>5} | {t_dense:>9.1f} {t_dlog:>9.1f} {gb / (t_dlog * 1e-6):>6.0f} | "
                      f"{t_block:>9.1f} {t_scan:>8.1f} {gbs:>6.0f} | {t_dense / t_block:>10.2f}x", flush=True)
                del q, qs, w, bt, out, cand, cand_in
                torch.cuda.empty_cache()
        del kv4
        torch.cuda.empty_cache()
    print(f"peak allocated {torch.cuda.max_memory_allocated() >> 20} MiB")


if __name__ == "__main__":
    main()
