#!/usr/bin/env python3
"""Exactness of the sm_120 sparse indexer path against the dense sm_120 path vLLM runs today.

Dense (today, per consumer layer):  DeepGEMM sm_120 fp8_fp4_paged_mqa_logits / fp8_fp4_mqa_logits (fp32 logits over the
    whole context) -> apply_candidate_mask (-inf outside the candidate blocks) -> det_topk_sm120 (decode / prefill form).
Sparse (this module):  candidate_blocks_to_sparse_indices (vLLM's Triton) -> get_*_sparse_mqa_logits_metadata ->
    fp8_fp4_(paged_)sparse_mqa_logits (fp32 at the candidate tokens only) -> det_topk_sm120 over the sparse columns ->
    vLLM's _sparse_topk_remap_kernel.

Checks per case:
  logits  every scored sparse column equals the dense logit at its token, bit for bit (int32 views); every column at
          or past the row's `end` is -inf.
  top-k   the final int32 [rows, 512] index rows are identical (positions, order, -1 fill).

Decode rows are flattened spec-decode rows (groups of 8 per request, context L - 7 + i), paged over a shared pool of
128-row pages; prefill rows are chunk tails of one or two requests in a packed workspace (the second at an unaligned
offset). Run inside the vllm-moet image on one GPU:
    python3 tests/test_exact.py [--quick] [--cases ref,decode,prefill,source,sbk16,block]

The `block` cases (tests/test_block_topk.py) are the dense layers without full-width logits: block_topk's top-k rows
and candidate blocks against the dense path, for the ratio-2 layers, the candidate source and the consumers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CBS, NCAND, PAGE_KV, TOPK, bits, ceil_to, dense_paged_logits, dense_prefill_logits, det_topk, expand,
    make_candidates, make_paged_cache, quant_view, rand_q, rand_scales, remap, sp,
)

DENSE_CHUNK_BYTES = 32 << 20
MAX_CHUNK_ROWS = 48
FAILS: list[str] = []


def check_logits(tag, logits_s, end, si, row_ks, dense, dense_row0, sbk):
    """Compare the sparse logits of rows [dense_row0, dense_row0 + dense.shape[0]) with the dense chunk."""
    rows = dense.shape[0]
    ls = logits_s[dense_row0:dense_row0 + rows]
    e = end[dense_row0:dense_row0 + rows].long()
    width = ls.shape[1]
    col = torch.arange(width, device="cuda")
    blocks = si[dense_row0:dense_row0 + rows].long()
    base = (row_ks[dense_row0:dense_row0 + rows].long() // sbk).unsqueeze(1)
    pos = (blocks.repeat_interleave(sbk, dim=1) - base) * sbk + (col % sbk)
    valid = col.unsqueeze(0) < e.unsqueeze(1)
    pos_c = torch.where(valid, pos, torch.zeros_like(pos)).clamp_(0, dense.shape[1] - 1)
    ref = torch.gather(dense, 1, pos_c)
    nvalid = int(valid.sum())
    mism = valid & (bits(ls) != bits(ref))
    nm = int(mism.sum())
    tail_bad = int((~valid & (ls != float("-inf"))).sum())
    if nm or tail_bad:
        r, c = mism.nonzero()[0].tolist() if nm else (0, 0)
        msg = (f"{tag}: logits {nm}/{nvalid} mismatched (first row {dense_row0 + r} col {c}: sparse {ls[r, c].item()!r}"
               f" dense {ref[r, c].item()!r}), {tail_bad} tail columns not -inf")
        FAILS.append(msg)
        print("  FAIL", msg)
    return nvalid, nm


def check_topk(tag, out_s, out_d, row0):
    neq = (out_s != out_d).any(dim=1)
    n = int(neq.sum())
    if n:
        r = int(neq.nonzero()[0])
        diff = (out_s[r] != out_d[r]).nonzero().flatten()[:4].tolist()
        msg = (f"{tag}: top-k rows differ: {n}/{out_s.shape[0]} (row {row0 + r}: cols {diff}: sparse "
               f"{out_s[r, diff].tolist()} dense {out_d[r, diff].tolist()})")
        FAILS.append(msg)
        print("  FAIL", msg)
    return n


# ---------------------------------------------------------------------------------------------------------------
# decode


def decode_case(ctx_target: int, rows: int, gen, pool_pages: int, kv3, cbs=CBS, ncand=NCAND, cands=None, tag=None,
                sbk=None):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import pick_sparse_block_kv

    t0 = time.time()
    sbk = sbk or pick_sparse_block_kv(cbs)
    tag = tag or f"decode ctx~{ctx_target} rows={rows}"
    kv4 = quant_view(kv3)
    nreq = (rows + 7) // 8
    jitter = torch.randint(0, max(1, ctx_target // 64), (nreq,), generator=torch.Generator().manual_seed(rows))
    req_len = [max(1, ctx_target - int(j)) for j in jitter]
    ctx = torch.tensor([max(0, req_len[r // 8] - 7 + r % 8) for r in range(rows)], dtype=torch.int32, device="cuda")
    max_ctx = int(ctx.max())
    npages = (max_ctx + PAGE_KV - 1) // PAGE_KV
    bt_req = torch.randint(0, pool_pages, (nreq, npages), dtype=torch.int32, device="cuda", generator=gen)
    bt = bt_req.repeat_interleave(8, dim=0)[:rows].contiguous()
    q, qs, w = rand_q(rows, gen)
    if cands is None:
        cands = make_candidates(ctx, gen, ncand, cbs)
    zeros = torch.zeros(rows, dtype=torch.int32, device="cuda")

    # sparse path, all rows at once
    si, end = expand(cands, zeros, ctx, cbs, sbk)
    meta = sp.get_paged_sparse_mqa_logits_metadata(ctx, bt, torch.arange(rows, device="cuda", dtype=torch.int32),
                                                   PAGE_KV, si, torch.int8, sbk)
    logits_s = sp.fp8_fp4_paged_sparse_mqa_logits((q.unsqueeze(1), qs.unsqueeze(1)), kv4, w, meta, si.shape[1], sbk)
    cols = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    sp.sparse_topk(logits_s, end, TOPK, cols)
    out_s = remap(cols, si, zeros, sbk, TOPK, logits_s.shape[1], torch.empty_like(cols))

    # dense path in row chunks
    width = ceil_to(max_ctx, 256)
    chunk = max(1, min(rows, MAX_CHUNK_ROWS, DENSE_CHUNK_BYTES // (width * 4)))
    nvalid = nm = ntop = 0
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        dense = dense_paged_logits(q[r0:r1], qs[r0:r1], kv4, w[r0:r1], ctx[r0:r1], bt[r0:r1], width)
        a, b = check_logits(tag, logits_s, end, si, zeros, dense, r0, sbk)
        nvalid += a
        nm += b
        apply_candidate_mask(dense, None, ctx[r0:r1], cands[r0:r1], cbs)
        out_d = torch.full((r1 - r0, TOPK), -1, dtype=torch.int32, device="cuda")
        det_topk().topk_decode(dense, ctx[r0:r1].view(-1, 1), 1, out_d, TOPK)
        ntop += check_topk(tag, out_s[r0:r1], out_d, r0)
        del dense
    torch.cuda.synchronize()
    status = "ok" if nm == 0 and ntop == 0 else "FAIL"
    print(f"{status:4s} {tag:44s} ctx {int(ctx.min())}..{max_ctx}  logits {nvalid - nm}/{nvalid} bit-equal  "
          f"top-k rows {rows - ntop}/{rows} equal  ({time.time() - t0:.1f}s)", flush=True)
    return cands, ctx, bt


def source_candidates(ctx, bt, kv3, gen):
    """Candidates exactly as the candidate-source layer writes them on sm_120: dense logits of another (q, w), then
    vLLM's select_candidate_blocks (step 31: det_topk over the block maxima, newest block pinned)."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import select_candidate_blocks

    rows = ctx.numel()
    q, qs, w = rand_q(rows, gen)
    width = ceil_to(int(ctx.max()), 256)
    dense = dense_paged_logits(q, qs, quant_view(kv3), w, ctx, bt, width)
    out = torch.full((rows, NCAND), -1, dtype=torch.int32, device="cuda")
    select_candidate_blocks(dense, None, ctx, NCAND, CBS, out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# prefill


def prefill_case(ctx_target: int, rows: int, gen, two_requests: bool, cbs=CBS, ncand=NCAND, sbk=None):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import pick_sparse_block_kv

    t0 = time.time()
    sbk = sbk or pick_sparse_block_kv(cbs)
    tag = f"prefill ctx~{ctx_target} rows={rows}{' 2req' if two_requests else ''}"
    # request A: [0, LA), request B (optional): [LA, LA + LB), LA odd so B's ks is unaligned
    if two_requests:
        LA = ctx_target // 3 * 2 + 3
        LB = ctx_target - LA + 5
        lens = [LA, LB]
        ra = rows // 2
        row_counts = [ra, rows - ra]
    else:
        lens = [ctx_target]
        row_counts = [rows]
    offs, T = [], 0
    for L in lens:
        offs.append(T)
        T += L
    ks_l, ke_l = [], []
    for off, L, n in zip(offs, lens, row_counts):
        for p in range(L - n, L):
            ks_l.append(off)
            ke_l.append(off + p + 1)
    cu_ks = torch.tensor(ks_l, dtype=torch.int32, device="cuda")
    cu_ke = torch.tensor(ke_l, dtype=torch.int32, device="cuda")
    kq = torch.randint(0, 256, (T, 64), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8).view(torch.int8)
    ksc = rand_scales(T, gen)
    q, qs, w = rand_q(rows, gen)
    local = cu_ke - cu_ks
    cands = make_candidates(local, gen, ncand, cbs)

    si, end = expand(cands, cu_ks, cu_ke, cbs, sbk)
    meta = sp.get_sparse_mqa_logits_metadata(cu_ks, cu_ke, T, si, torch.int8, sbk, use_unaligned_ks=True)
    logits_s = sp.fp8_fp4_sparse_mqa_logits((q, qs), (kq, ksc), w, meta, si.shape[1], sbk, use_unaligned_ks=True)
    cols = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    sp.sparse_topk(logits_s, end, TOPK, cols)
    out_s = remap(cols, si, cu_ks, sbk, TOPK, logits_s.shape[1], torch.empty_like(cols))

    max_k = int(local.max())
    width = ceil_to(max_k, 256)
    chunk = max(1, min(rows, MAX_CHUNK_ROWS, DENSE_CHUNK_BYTES // (width * 4)))
    nvalid = nm = ntop = 0
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        dense = dense_prefill_logits(q[r0:r1], qs[r0:r1], kq, ksc, w[r0:r1], cu_ks[r0:r1], cu_ke[r0:r1], max_k)
        # dense logits are compressed (column 0 = ks): positions are request-local, as the remap's
        a, b = check_logits(tag, logits_s, end, si, cu_ks, dense, r0, sbk)
        nvalid += a
        nm += b
        row_ends = (cu_ke[r0:r1] - cu_ks[r0:r1]).contiguous()
        apply_candidate_mask(dense, None, row_ends, cands[r0:r1], cbs)
        out_d = torch.full((r1 - r0, TOPK), -1, dtype=torch.int32, device="cuda")
        det_topk().topk_prefill(dense, None, row_ends, out_d, r1 - r0, TOPK, True)
        ntop += check_topk(tag, out_s[r0:r1], out_d, r0)
        del dense
    torch.cuda.synchronize()
    status = "ok" if nm == 0 and ntop == 0 else "FAIL"
    print(f"{status:4s} {tag:44s} ks%8={sorted(set(o % 8 for o in offs))}  logits {nvalid - nm}/{nvalid} bit-equal  "
          f"top-k rows {rows - ntop}/{rows} equal  ({time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------


def torch_reference_check(gen):
    """Sanity: the dense kernel (the oracle) matches a float64 dequantized reference to fp32 rounding on a small case,
    so both paths are not equal-but-wrong (e.g. a misread page layout)."""
    rows, L = 3, 300
    kv3 = make_paged_cache(8, gen)
    kv4 = quant_view(kv3)
    bt = torch.randperm(8, device="cuda", generator=gen)[:3].to(torch.int32).repeat(rows, 1).contiguous()
    ctx = torch.full((rows,), L, dtype=torch.int32, device="cuda")
    q, qs, w = rand_q(rows, gen)
    dense = dense_paged_logits(q, qs, kv4, w, ctx, bt, 512)

    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64,
                        device="cuda")

    def deq(vals_u8, sc_u8):  # [N, 64] bytes, [N, 4] exps -> [N, 128]
        lo = e2m1[(vals_u8 & 0xF).long()]
        hi = e2m1[(vals_u8 >> 4).long()]
        x = torch.stack([lo, hi], dim=-1).reshape(vals_u8.shape[0], 128)
        s = torch.pow(2.0, sc_u8.double() - 127).repeat_interleave(32, dim=1)
        return x * s

    flat = kv3.view(8, -1)
    toks = torch.arange(L, device="cuda")
    pages = bt[0, (toks // PAGE_KV).long()].long()
    off = toks % PAGE_KV
    vals = flat[pages.unsqueeze(1), (off * 64).unsqueeze(1) + torch.arange(64, device="cuda")]
    scs = flat[pages.unsqueeze(1), (PAGE_KV * 64 + off * 4).unsqueeze(1) + torch.arange(4, device="cuda")]
    k = deq(vals, scs)  # [L, 128]
    qd = deq(q.view(torch.uint8).reshape(-1, 64), qs.view(torch.uint8).view(-1, 4)).view(rows, 32, 128)
    ref = (torch.relu(torch.einsum("rhd,ld->rhl", qd, k)) * w.double().unsqueeze(-1)).sum(1)
    err = (dense[:, :L].double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    ok = err <= 1e-5 * max(scale, 1.0)
    print(f"{'ok' if ok else 'FAIL':4s} dense kernel vs float64 reference: max |err| {err:.3e} (max |ref| {scale:.3e})")
    if not ok:
        FAILS.append("dense kernel disagrees with the float64 reference: page layout misunderstood?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="ref,decode,prefill,source,sbk16,block")
    ap.add_argument("--pool-pages", type=int, default=4096, help="shared KV pool pages (4096 = 36 MB)")
    args = ap.parse_args()
    cases = set(args.cases.split(","))
    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}, "
          f"free {torch.cuda.mem_get_info()[0] >> 20} MiB", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(1234)

    if "ref" in cases:
        torch_reference_check(gen)

    ctxs = [4096, 65536, 262144, 1048576]
    rows_list = [1, 8, 37, 128, 512]
    if args.quick:
        ctxs, rows_list = [4096, 65536], [1, 37]

    if cases & {"decode", "source", "sbk16"}:
        kv3 = make_paged_cache(args.pool_pages, gen)
    if "decode" in cases:
        # short contexts: every block a candidate (incl. contexts below the top-k)
        for L, r in [(100, 8), (513, 8), (5000, 16), (16380, 8), (16400, 8)]:
            decode_case(L, r, gen, args.pool_pages, kv3)
        for L in ctxs:
            for r in rows_list:
                decode_case(L, r, gen, args.pool_pages, kv3)
    if "source" in cases:
        for L, r in ([(65536, 64), (262144, 16)] if not args.quick else [(65536, 16)]):
            nreq = (r + 7) // 8
            ctx = torch.tensor([max(0, L - 7 + i % 8) for i in range(r)], dtype=torch.int32, device="cuda")
            npages = (L + PAGE_KV) // PAGE_KV
            bt = torch.randint(0, args.pool_pages, (nreq, npages), dtype=torch.int32, device="cuda",
                               generator=gen).repeat_interleave(8, dim=0)[:r].contiguous()
            cands = source_candidates(ctx, bt, kv3, gen)
            nvalid = int((cands >= 0).sum(1).min())
            print(f"     source candidates: {nvalid}..{int((cands >= 0).sum(1).max())} per row, ascending="
                  f"{bool((cands[:, 1:nvalid] > cands[:, :nvalid - 1]).all())}")
            # same rows/bt/ctx: rebuild a decode case around these candidates
            decode_case_fixed(ctx, bt, cands, gen, kv3, f"decode ctx={L} rows={r} source-cands")
    if "sbk16" in cases:
        decode_case(65536, 37, gen, args.pool_pages, kv3, cbs=16, ncand=1024, tag="decode cbs=16 (sbk 16) rows=37")
        decode_case(65536, 37, gen, args.pool_pages, kv3, cbs=32, ncand=512, tag="decode cbs=32 (sbk 16 x2) rows=37")
        prefill_case(65536, 37, gen, True, cbs=16, ncand=1024)
    if cases & {"decode", "source", "sbk16"}:
        del kv3
        torch.cuda.empty_cache()
    if "prefill" in cases:
        for L, r in [(300, 64), (5000, 64)]:
            prefill_case(L, r, gen, True)
        for L in ctxs:
            for r in rows_list:
                prefill_case(L, r, gen, two_requests=(r >= 8))

    if "block" in cases:
        import test_block_topk

        test_block_topk.run(args.quick, {"decode", "mixed", "ties", "wild", "prefill"})
        FAILS.extend(test_block_topk.FAILS)

    free, total = torch.cuda.mem_get_info()
    print(f"peak allocated {torch.cuda.max_memory_allocated() >> 20} MiB, reserved now "
          f"{torch.cuda.memory_reserved() >> 20} MiB, device used now {(total - free) >> 20} MiB")
    if FAILS:
        print(f"\n{len(FAILS)} FAILURES")
        sys.exit(1)
    print("\nALL EXACT")


def decode_case_fixed(ctx, bt, cands, gen, kv3, tag):
    """decode_case with given rows / pages / candidates."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask

    t0 = time.time()
    rows = ctx.numel()
    kv4 = quant_view(kv3)
    q, qs, w = rand_q(rows, gen)
    zeros = torch.zeros(rows, dtype=torch.int32, device="cuda")
    si, end = expand(cands, zeros, ctx, CBS, 8)
    meta = sp.get_paged_sparse_mqa_logits_metadata(ctx, bt, None, PAGE_KV, si, torch.int8, 8)
    logits_s = sp.fp8_fp4_paged_sparse_mqa_logits((q.unsqueeze(1), qs.unsqueeze(1)), kv4, w, meta, si.shape[1], 8)
    cols = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    sp.sparse_topk(logits_s, end, TOPK, cols)
    out_s = remap(cols, si, zeros, 8, TOPK, logits_s.shape[1], torch.empty_like(cols))
    width = ceil_to(int(ctx.max()), 256)
    chunk = max(1, min(rows, MAX_CHUNK_ROWS, DENSE_CHUNK_BYTES // (width * 4)))
    nvalid = nm = ntop = 0
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        dense = dense_paged_logits(q[r0:r1], qs[r0:r1], kv4, w[r0:r1], ctx[r0:r1], bt[r0:r1], width)
        a, b = check_logits(tag, logits_s, end, si, zeros, dense, r0, 8)
        nvalid += a
        nm += b
        apply_candidate_mask(dense, None, ctx[r0:r1], cands[r0:r1], CBS)
        out_d = torch.full((r1 - r0, TOPK), -1, dtype=torch.int32, device="cuda")
        det_topk().topk_decode(dense, ctx[r0:r1].view(-1, 1), 1, out_d, TOPK)
        ntop += check_topk(tag, out_s[r0:r1], out_d, r0)
    status = "ok" if nm == 0 and ntop == 0 else "FAIL"
    print(f"{status:4s} {tag:44s} logits {nvalid - nm}/{nvalid} bit-equal  top-k rows {rows - ntop}/{rows} equal  "
          f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
