#!/usr/bin/env python3
"""Exactness of block_topk (no full-width logits) against the dense sm_120 indexer path vLLM runs today.

Dense (today), per layer call:
  plain     (layers 2, 8, 14: ratio 2, 64-row pages)  DeepGEMM fp8_fp4_(paged_)mqa_logits -> det_topk
  source    (layer 20: 128-row pages)                 logits -> select_candidate_blocks (step 31) -> det_topk
  consumer  (layers 24..36)                           logits -> apply_candidate_mask -> det_topk
Block (this module): block_topk.paged_topk / contiguous_topk (plain, source), *_candidates_topk (consumer).

Checked per case: the int32 [rows, 512] top-k rows are identical (positions, order, -1 fill), and for the source the
int32 [rows, 2048] candidate blocks are identical.

Decode rows are flattened spec-decode rows (groups of 8 per request, context L - 7 + i) over a shared page pool; the
block path runs at the engine's static width (1,048,576 for the ratio-1 layers, 524,288 for ratio 2), the dense
reference in row chunks at the context's width. Prefill rows are chunk tails of one or two requests in a packed
workspace (the second at an unaligned offset). Ties: a pool of 2-3 pages (every key repeated thousands of times) and a
constant cache (all logits of a row equal). Run inside the vllm-moet image on one GPU:
    python3 tests/test_block_topk.py [--quick] [--cases decode,prefill,ties,mixed,wild]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CBS, HEAD_BYTES, NCAND, ROWB, TOPK, ceil_to, dense_prefill_logits, det_topk, num_sms, quant_view, rand_q,
    rand_scales,
)
from sm120_sparse_indexer import block_topk as bk  # noqa: E402

DENSE_CHUNK_BYTES = 48 << 20
FAILS: list[str] = []
KINDS = {"plain": (64, 1 << 19), "source": (128, 1 << 20), "consumer": (128, 1 << 20)}  # page_kv, engine width


def make_cache(num_pages: int, page_kv: int, gen, const: bool = False, scale_range=(122, 132)) -> torch.Tensor:
    """[num_pages, page_kv, 68] uint8: per page page_kv x 64 B of e2m1 pairs, then page_kv x 4 UE8M0 bytes."""
    kv = torch.empty(num_pages, page_kv, HEAD_BYTES, dtype=torch.uint8, device="cuda")
    flat = kv.view(num_pages, page_kv * HEAD_BYTES)
    if const:
        row = torch.randint(0, 256, (ROWB,), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8)
        flat[:, : page_kv * ROWB] = row.repeat(page_kv)
        flat[:, page_kv * ROWB:] = rand_scales(1, gen).view(torch.uint8).repeat(page_kv)
        return kv
    flat[:, : page_kv * ROWB] = torch.randint(0, 256, (num_pages, page_kv * ROWB), dtype=torch.int32, device="cuda",
                                              generator=gen).to(torch.uint8)
    flat[:, page_kv * ROWB:] = rand_scales(num_pages * page_kv, gen, *scale_range).view(torch.uint8).view(
        num_pages, page_kv * 4)
    return kv


def dense_paged(q, qs, kv4, w, ctx, bt, width: int, page_kv: int) -> torch.Tensor:
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata

    c2 = ctx.view(-1, 1)
    sched = get_paged_mqa_logits_metadata(c2, page_kv, num_sms())
    return fp8_fp4_paged_mqa_logits((q.unsqueeze(1), qs.unsqueeze(1)), kv4, w, c2, bt, sched, max_model_len=width)


def compare(tag: str, what: str, mine: torch.Tensor, ref: torch.Tensor) -> int:
    neq = (mine != ref).any(dim=1)
    n = int(neq.sum())
    if n:
        r = int(neq.nonzero()[0])
        d = (mine[r] != ref[r]).nonzero().flatten()[:4].tolist()
        msg = (f"{tag}: {what} rows differ: {n}/{mine.shape[0]} (row {r}: cols {d}: block {mine[r, d].tolist()} dense "
               f"{ref[r, d].tolist()})")
        FAILS.append(msg)
        print("  FAIL", msg, flush=True)
    return n


def decode_rows(ctx_list, page_kv: int, pool_pages: int, gen):
    """ctx (one entry per row; rows of a request = a run of up to 8 consecutive entries share its pages), bt."""
    rows = len(ctx_list)
    ctx = torch.tensor(ctx_list, dtype=torch.int32, device="cuda")
    npages = max(1, (max(ctx_list) + page_kv - 1) // page_kv)
    nreq = (rows + 7) // 8
    bt = torch.randint(0, pool_pages, (nreq, npages), dtype=torch.int32, device="cuda",
                       generator=gen).repeat_interleave(8, dim=0)[:rows].contiguous()
    return ctx, bt


def spec_rows(ctx_target: int, rows: int):
    nreq = (rows + 7) // 8
    jitter = torch.randint(0, max(1, ctx_target // 64), (nreq,), generator=torch.Generator().manual_seed(rows))
    req_len = [max(1, ctx_target - int(j)) for j in jitter]
    return [max(0, req_len[r // 8] - 7 + r % 8) for r in range(rows)]


def decode_case(kind: str, ctx_list, kv3, gen, tag: str, check: bool = True, count_nonfinite: bool = False,
                group: int = 8, force_group: bool = False) -> int:
    """Returns the number of differing rows (top-k + candidates). check=False: counted and printed, not a failure.
    group: the scan's rows per request (a traversal order only: every value must give the same rows); force_group:
    grouped even when the launch's K is small (by default such launches go row by row)."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask, select_candidate_blocks

    t0 = time.time()
    bk.set_decode_group(group)
    bk._ext().set_scan_group_min_units(0 if force_group else 65536)
    page_kv, width_engine = KINDS[kind]
    kv4 = quant_view(kv3)
    rows = len(ctx_list)
    ctx, bt = decode_rows(ctx_list, page_kv, kv3.shape[0], gen)
    q, qs, w = rand_q(rows, gen)
    max_ctx = max(1, int(ctx.max()))
    width = ceil_to(max_ctx, 256)
    chunk = max(1, min(rows, DENSE_CHUNK_BYTES // (width * 4)))
    out = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
    ref = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    cand = cand_ref = None
    if kind == "consumer":
        # the candidates the source layer wrote for these rows (dense, another q / w)
        q2, qs2, w2 = rand_q(rows, gen)
        cand_in = torch.full((rows, NCAND), -1, dtype=torch.int32, device="cuda")
        for r0 in range(0, rows, chunk):
            sl = slice(r0, min(rows, r0 + chunk))
            dense = dense_paged(q2[sl], qs2[sl], kv4, w2[sl], ctx[sl], bt[sl], width, page_kv)
            select_candidate_blocks(dense, None, ctx[sl], NCAND, CBS, cand_in[sl])
            del dense
        bk.paged_candidates_topk(q.unsqueeze(1), qs.unsqueeze(1), kv4, w, ctx, bt, width_engine, TOPK, out, cand_in)
    else:
        if kind == "source":
            cand = torch.full((rows, NCAND), -7, dtype=torch.int32, device="cuda")
            cand_ref = torch.full((rows, NCAND), -1, dtype=torch.int32, device="cuda")
        bk.paged_topk(q.unsqueeze(1), qs.unsqueeze(1), kv4, w, ctx, bt, width_engine, TOPK, out, candidates=cand)
    nonfinite = 0
    for r0 in range(0, rows, chunk):
        sl = slice(r0, min(rows, r0 + chunk))
        dense = dense_paged(q[sl], qs[sl], kv4, w[sl], ctx[sl], bt[sl], width, page_kv)
        if count_nonfinite:
            inside = torch.arange(dense.shape[1], device="cuda").unsqueeze(0) < ctx[sl].unsqueeze(1)
            nonfinite += int((inside & ~torch.isfinite(dense)).sum())
        if kind == "source":
            select_candidate_blocks(dense, None, ctx[sl], NCAND, CBS, cand_ref[sl])
        elif kind == "consumer":
            apply_candidate_mask(dense, None, ctx[sl], cand_in[sl], CBS)
        det_topk().topk_decode(dense, ctx[sl].view(-1, 1), 1, ref[sl], TOPK)
        del dense
    torch.cuda.synchronize()
    extra = f"ctx {int(ctx.min())}..{max_ctx}" + (f" ({nonfinite} inf/NaN logits)" if count_nonfinite else "")
    return report(tag, kind, rows, out, ref, cand, cand_ref, extra, t0, check)


def report(tag, kind, rows, out, ref, cand, cand_ref, extra, t0, check=True) -> int:
    fails_before = len(FAILS)
    bad = compare(tag, "top-k", out, ref)
    cbad = compare(tag, "candidate", cand, cand_ref) if cand is not None else 0
    if not check:
        del FAILS[fails_before:]
    status = "ok" if bad + cbad == 0 else ("FAIL" if check else "diff")
    cmsg = f"  candidates {rows - cbad}/{rows}" if cand is not None else ""
    print(f"{status:4s} {tag:46s} {extra}  top-k rows {rows - bad}/{rows} equal{cmsg}  ({time.time() - t0:.1f}s)",
          flush=True)
    return bad + cbad


def prefill_case(kind: str, lens, row_counts, gen, tag: str) -> int:
    """Requests of lengths `lens` packed into one workspace; the last row_counts[i] positions of request i are rows."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask, select_candidate_blocks

    t0 = time.time()
    offs, T = [], 0
    for L in lens:
        offs.append(T)
        T += L
    ks_l, ke_l = [], []
    for off, L, n in zip(offs, lens, row_counts):
        for pos in range(L - n, L):
            ks_l.append(off)
            ke_l.append(off + pos + 1)
    rows = len(ks_l)
    cu_ks = torch.tensor(ks_l, dtype=torch.int32, device="cuda")
    cu_ke = torch.tensor(ke_l, dtype=torch.int32, device="cuda")
    kq = torch.randint(0, 256, (T, 64), dtype=torch.int32, device="cuda", generator=gen).to(torch.uint8).view(torch.int8)
    ksc = rand_scales(T, gen)
    q, qs, w = rand_q(rows, gen)
    row_ends = (cu_ke - cu_ks).contiguous()
    max_k = int(row_ends.max())
    chunk = max(1, min(rows, DENSE_CHUNK_BYTES // (ceil_to(max_k, 256) * 4)))
    out = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
    ref = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    cand = cand_ref = None

    def dense_chunk(qq, qqs, ww, sl):
        return dense_prefill_logits(qq[sl], qqs[sl], kq, ksc, ww[sl], cu_ks[sl], cu_ke[sl], max_k)

    if kind == "consumer":
        q2, qs2, w2 = rand_q(rows, gen)
        cand_in = torch.full((rows, NCAND), -1, dtype=torch.int32, device="cuda")
        for r0 in range(0, rows, chunk):
            sl = slice(r0, min(rows, r0 + chunk))
            dense = dense_chunk(q2, qs2, w2, sl)
            select_candidate_blocks(dense, None, row_ends[sl], NCAND, CBS, cand_in[sl])
            del dense
        bk.contiguous_candidates_topk(q, qs, kq, ksc, w, cu_ks, cu_ke, max_k, TOPK, out, cand_in)
    else:
        if kind == "source":
            cand = torch.full((rows, NCAND), -7, dtype=torch.int32, device="cuda")
            cand_ref = torch.full((rows, NCAND), -1, dtype=torch.int32, device="cuda")
        bk.contiguous_topk(q, qs, kq, ksc, w, cu_ks, cu_ke, max_k, TOPK, out, candidates=cand)
    for r0 in range(0, rows, chunk):
        sl = slice(r0, min(rows, r0 + chunk))
        dense = dense_chunk(q, qs, w, sl)
        if kind == "source":
            select_candidate_blocks(dense, None, row_ends[sl], NCAND, CBS, cand_ref[sl])
        elif kind == "consumer":
            apply_candidate_mask(dense, None, row_ends[sl], cand_in[sl], CBS)
        det_topk().topk_prefill(dense, None, row_ends[sl], ref[sl], sl.stop - sl.start, TOPK, True)
        del dense
    torch.cuda.synchronize()
    return report(tag, kind, rows, out, ref, cand, cand_ref, f"ks%8={sorted(set(o % 8 for o in offs))} max_k {max_k}", t0)


def run(quick: bool, cases: set[str], pool_mib: float = 36.0) -> None:
    gen = torch.Generator(device="cuda").manual_seed(4321)
    print(f"block_topk: kernel info [scan regs, scan CTAs/SM, rescore regs, rescore CTAs/SM] "
          f"{bk._ext().kernel_info(0)}", flush=True)
    for kind, (page_kv, width) in KINDS.items():  # the engine's registration: one scratch size per decode call
        if kind != "consumer":
            bk.reserve_paged(width, NCAND if kind == "source" else TOPK, TOPK, kind == "source")
    ctxs = [4096, 65536, 262144, 1048576]
    rows_list = [1, 8, 37, 128, 512]
    if quick:
        ctxs, rows_list = [4096, 65536], [1, 37]
    kinds = list(KINDS)

    if cases & {"decode", "mixed", "ties", "wild"}:
        pools = {pk: make_cache(int(pool_mib * 2**20 / (pk * HEAD_BYTES)), pk, gen) for pk in (64, 128)}
    if "decode" in cases:
        for kind in kinds:
            ratio = 2 if kind == "plain" else 1
            kv3 = pools[KINDS[kind][0]]
            # short contexts around the top-k, the block count and the single-pass thresholds (16,384 / 65,536)
            for L in [100, 513, 4090, 4097, 5000, 16380, 16385, 16400, 65530, 65537, 65560]:
                decode_case(kind, spec_rows(L, 8), kv3, gen, f"decode {kind} ctx~{L} rows=8")
            for L in ctxs:
                for r in rows_list:
                    decode_case(kind, spec_rows(L // ratio, r), kv3, gen, f"decode {kind} ctx~{L // ratio} rows={r}")
                decode_case(kind, spec_rows(L // ratio, 37), kv3, gen, f"decode {kind} ctx~{L // ratio} rows=37 group=1",
                            group=1)
                decode_case(kind, spec_rows(L // ratio, 37), kv3, gen,
                            f"decode {kind} ctx~{L // ratio} rows=37 grouped", force_group=True)
    if "mixed" in cases:
        # one batch of very different rows: empty, below the top-k, around the thresholds, long
        base = [0, 1, 7, 8, 9, 100, 511, 512, 513, 4095, 4096, 4097, 0, 16383, 16384, 16385, 16391, 16392, 16393, 70000,
                5, 200000, 0, 0, 33000, 4104, 4089, 131071, 131072, 131073, 3, 0]
        for kind in kinds:
            ratio = 2 if kind == "plain" else 1
            lst = [c // ratio if c > 20000 else c for c in base]
            for grp in (1, 3, 8, 64):  # the scan's row groups: a traversal order, never a different result
                decode_case(kind, lst, pools[KINDS[kind][0]], gen, f"decode {kind} mixed rows={len(lst)} group={grp}",
                            group=grp, force_group=True)
            if not quick:
                big = (lst * 16)[:500] + [1000000 // ratio, 0, 999999 // ratio]
                for grp in (1, 8):
                    decode_case(kind, big, pools[KINDS[kind][0]], gen,
                                f"decode {kind} mixed rows={len(big)} group={grp}", group=grp)
    if "ties" in cases:
        for kind in kinds:
            page_kv = KINDS[kind][0]
            ratio = 2 if kind == "plain" else 1
            for npages, name in [(2, "2-page pool"), (3, "3-page pool")]:
                kv3 = make_cache(npages, page_kv, gen)
                for L, r in ([(65536, 37)] if quick else [(4096, 16), (65536, 37), (262144, 64), (1048576, 16)]):
                    decode_case(kind, spec_rows(L // ratio, r), kv3, gen, f"decode {kind} ties {name} ctx~{L // ratio} rows={r}")
            kv3 = make_cache(4, page_kv, gen, const=True)
            for L, r in ([(65536, 16)] if quick else [(600, 8), (65536, 16), (262144, 16)]):
                decode_case(kind, spec_rows(L // ratio, r), kv3, gen, f"decode {kind} ties constant keys ctx~{L // ratio} rows={r}")
    if "wild" in cases:
        # UE8M0 scales over the whole byte range: +-inf and NaN logits. The plain and source layers stay identical;
        # a consumer differs from the dense path where its row has fewer than 512 candidate tokens above -inf (the
        # dense top-k then fills up with masked columns) - reported, not a failure.
        for kind in kinds:
            page_kv = KINDS[kind][0]
            ratio = 2 if kind == "plain" else 1
            kv3 = make_cache(64, page_kv, gen, scale_range=(0, 255))
            for L, r in [(4096, 16), (65536, 24)]:
                decode_case(kind, spec_rows(L // ratio, r), kv3, gen, f"decode {kind} inf/NaN logits ctx~{L // ratio} rows={r}",
                            check=kind != "consumer", count_nonfinite=True)
    if cases & {"decode", "mixed", "ties", "wild"}:
        del pools
        torch.cuda.empty_cache()
    if "prefill" in cases:
        for kind in kinds:
            for L, r in [(300, 64), (5000, 64), (16400, 64)]:
                prefill_case(kind, [L // 3 * 2 + 3, L - (L // 3 * 2 + 3) + 5], [r // 2, r - r // 2], gen,
                             f"prefill {kind} ctx~{L} rows={r} 2req")
            # around the plain layers' single-pass width (65,536 columns; all rows short up to 74,898)
            for lens_, rows_ in [([65540], [40]), ([70000], [64]), ([74900], [40]), ([80000], [64]),
                                 ([90000, 30000], [32, 32])]:
                prefill_case(kind, lens_, rows_, gen, f"prefill {kind} lens={lens_} rows={sum(rows_)}")
            for L in ctxs:
                for r in rows_list:
                    if r >= 8:
                        la = L // 3 * 2 + 3
                        prefill_case(kind, [la, L - la + 5], [r // 2, r - r // 2], gen, f"prefill {kind} ctx~{L} rows={r} 2req")
                    else:
                        prefill_case(kind, [L], [r], gen, f"prefill {kind} ctx~{L} rows={r}")
            if not quick:
                # more rows than one scan launch takes (8192): the host splits the batch
                prefill_case(kind, [9500], [9000], gen, f"prefill {kind} ctx~9500 rows=9000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="decode,mixed,ties,wild,prefill")
    ap.add_argument("--pool-mib", type=float, default=36.0)
    args = ap.parse_args()
    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}, "
          f"free {torch.cuda.mem_get_info()[0] >> 20} MiB", flush=True)
    run(args.quick, set(args.cases.split(",")), args.pool_mib)
    print(f"peak allocated {torch.cuda.max_memory_allocated() >> 20} MiB")
    if FAILS:
        print(f"\n{len(FAILS)} FAILURES")
        sys.exit(1)
    print("\nALL EXACT")


if __name__ == "__main__":
    main()
