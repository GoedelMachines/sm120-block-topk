#!/usr/bin/env python3
"""Scan / whole-call time of block_topk.paged_topk for a few launch shapes (threads per CTA, CTAs per SM)."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HEAD_BYTES, NCAND, TOPK, quant_view, rand_q
from sm120_sparse_indexer import block_topk as bk
from bench_block_topk import graph_time, make_cache

torch.cuda.init()
gen = torch.Generator(device="cuda").manual_seed(7)
ext = bk._ext()
cfgs = [(256, 0), (256, 2), (128, 0), (128, 4), (64, 0), (32, 0)]
shapes = [("ratio2", 32768, 8), ("ratio2", 32768, 128), ("source", 65536, 8), ("source", 65536, 128), ("source", 1 << 20, 128)]
caches = {}
RESCORE = None
print("kernel_info [scan regs, scan CTAs/SM, rescore regs, rescore CTAs/SM]:", ext.kernel_info(0))
print("shape".ljust(26) + "".join(f"{str(c):>16}" for c in cfgs))
for kind, L, rows in shapes:
    source = kind == "source"
    page_kv = 128 if source else 64
    if page_kv not in caches:
        caches[page_kv] = quant_view(make_cache(int(46 * 2**20 / (page_kv * HEAD_BYTES)), page_kv, gen))
    kv4 = caches[page_kv]
    pool = kv4.shape[0]
    W = (1 << 20) if source else (1 << 19)
    nreq = (rows + 7) // 8
    ctx = torch.tensor([L - 7 + r % 8 for r in range(rows)], dtype=torch.int32, device="cuda")
    bt = torch.randint(0, pool, (nreq, (L + page_kv) // page_kv), dtype=torch.int32, device="cuda",
                       generator=gen).repeat_interleave(8, dim=0)[:rows].contiguous()
    q, qs, w = rand_q(rows, gen)
    q4, qs4 = q.unsqueeze(1), qs.unsqueeze(1)
    out = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    cand = torch.empty(rows, NCAND, dtype=torch.int32, device="cuda") if source else None
    line = f"{kind} {L}x{rows}".ljust(26)
    for thr, per in cfgs:
        ext.set_launch(thr, per, 64, 0, 8)
        bk._STOP_AFTER = 1
        t_scan = graph_time(lambda: bk.paged_topk(q4, qs4, kv4, w, ctx, bt, W, TOPK, out, candidates=cand), 30)
        bk._STOP_AFTER = 0
        t_all = graph_time(lambda: bk.paged_topk(q4, qs4, kv4, w, ctx, bt, W, TOPK, out, candidates=cand), 30)
        line += f"{t_scan:>8.1f}/{t_all:<7.1f}"
    print(line, flush=True)
