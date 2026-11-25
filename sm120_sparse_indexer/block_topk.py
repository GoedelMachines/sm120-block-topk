"""Exact indexer top-k on sm_120 without full-width logits (the dense indexer layers of DeepSeek-V4.1).

Today's dense path, per indexer layer call: DeepGEMM's sm_120 MQA logits fp32 ``[rows, width]`` (width = max_model_len /
compress_ratio in decode: 4 MiB per row at 1M), then ``det_topk_sm120`` over them; the candidate source (layer 20) also
takes the block maxima's top 2048 as candidate blocks. This module returns the same int32 top-k rows (and the same
candidate blocks), bit for bit, from fp32 block maxima ``[rows, width / 8]`` and the logits of the selected blocks
``[rows, S * 8]``:

    scan      every token scored once; long rows keep each 8-token block's maximum, short rows keep their logits (so
              a short row costs one pass, as in the dense path)
    det_topk  the S blocks with the largest maxima (ties to the lower block), ascending
    rescore   the tokens of those blocks scored again (column j * 8 + o = token sel[j] * 8 + o)
    det_topk  the top-k columns (ties to the lower column = the lower token)
    finish    columns -> tokens; the candidate blocks' -inf -> -1 rule

S = topk blocks (512) for a plain layer; for the candidate source S = the candidate count (2048): its top-512 tokens lie
in its top-512 blocks, which the pinned top 2048 contain. Why this is exact is at the top of csrc/block_topk_sm120.cu.

    paged_topk(q, q_scale, kv_cache, weights, context_lens, block_table, width, topk, out, candidates=None)
    contiguous_topk(q, q_scale, kv, kv_scale, weights, cu_ks, cu_ke, width, topk, out, candidates=None)
    paged_candidates_topk(q, q_scale, kv_cache, weights, context_lens, block_table, width, topk, out, candidates)
    contiguous_candidates_topk(q, q_scale, kv, kv_scale, weights, cu_ks, cu_ke, width, topk, out, candidates)
    reserve_paged(width, num_blocks, topk, source)   # at layer construction: one scratch size for every paged call
    set_decode_group(rows_per_request)               # decode rows that share their pages (1 + speculative tokens)

The ``*_candidates_topk`` forms are a candidate consumer's (layers 24..36): the dense path computes full-width logits
and masks every token outside the candidate blocks to -inf before its top-k; here only the candidate blocks' tokens are
scored (compact -> rescore -> det_topk -> finish). Identical whenever the row has 512 candidate tokens above -inf
(always, with finite logits); the dense path would otherwise fill up with the row's lowest -inf columns.

``out`` int32 ``[rows, >= topk]`` gets what ``det_topk_sm120.topk_decode`` / ``topk_prefill(relative=True)`` write for
the dense logits: the top-k positions ascending, then -1. ``candidates`` int32 ``[rows, K]`` gets what vLLM's
``select_candidate_blocks`` (VLLM_MOET_CANDIDATE_DET_TOPK=1) writes.

A row is short when it has at most ``short`` logit columns: the candidate source 16,384 (its S blocks), a plain layer
65,536 (its scratch has the room: half the block maxima of the source). $VLLM_MOET_INDEXER_BLOCK_TOPK_SHORT_SOURCE /
$VLLM_MOET_INDEXER_BLOCK_TOPK_SHORT move them (more columns: fewer two-pass rows, 4 bytes per column and row more).

Scratch: one allocation per call, ``rows * (width / 8 + short + S + topk) * 4`` bytes (0.57 MiB per row for the
source at width 1,048,576, 0.50 for a plain layer at 524,288; the dense path: 4.5 and 2 MiB). ``reserve_paged`` makes
every paged call ask for the largest registered layer's size, so under a CUDA graph the calls of one step reuse one
block of the graph pool.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from . import _det_topk, _load_ext

EXT_NAME = "vllm_moet_block_topk_sm120"
SRC = Path(__file__).resolve().parent / "csrc" / "block_topk_sm120.cu"
BLOCK = 8
# Logit columns of a short (single-pass) row, at least S * 8: [plain layer, candidate source].
SHORT_COLS = [int(os.environ.get("VLLM_MOET_INDEXER_BLOCK_TOPK_SHORT", "65536")),
              int(os.environ.get("VLLM_MOET_INDEXER_BLOCK_TOPK_SHORT_SOURCE", "16384"))]

__all__ = ["paged_topk", "contiguous_topk", "paged_candidates_topk", "contiguous_candidates_topk", "reserve_paged",
           "set_decode_group",
           "scratch_words_per_row", "paged_supported", "contiguous_supported", "available"]


@functools.lru_cache(maxsize=None)
def _ext():
    return _load_ext(EXT_NAME, SRC, EXT_NAME)


def available() -> bool:
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            return False
        _ext()
        _det_topk()
        return True
    except Exception:  # noqa: BLE001
        return False


def _up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _layout(width: int, num_blocks: int, topk: int, source: bool) -> tuple[int, int, int, int]:
    """(short blocks, score columns, logit columns, int32 words per row) of a call; columns are multiples of 4."""
    wb = (width + BLOCK - 1) // BLOCK
    short_blocks = max(num_blocks, SHORT_COLS[source] // BLOCK)
    # block maxima + the short columns take width / 2 + 4 * short bytes per row: when plain logits of the whole
    # width are no larger, every row is short (the dense path's footprint, and no second pass)
    if not source and wb * BLOCK * 7 <= short_blocks * BLOCK * 8:
        short_blocks = max(short_blocks, wb)
    all_short = wb <= short_blocks
    score_cols = 0 if all_short and not source else _up(wb, 4)  # no block maxima when no row is long
    logit_cols = _up(width, 16) if all_short else short_blocks * BLOCK
    return short_blocks, score_cols, logit_cols, score_cols + logit_cols + _up(num_blocks, 4) + _up(topk, 4) + 3


def scratch_words_per_row(width: int, num_blocks: int, topk: int, source: bool = False) -> int:
    return _layout(width, num_blocks, topk, source)[3]


_PAGED_ROW_WORDS = 0
_DECODE_GROUP = 1
_STOP_AFTER = 0  # bench only: return after step n (1 scan, 2 block top-k, 3 rescore, 4 column top-k)


def reserve_paged(width: int, num_blocks: int, topk: int, source: bool = False) -> None:
    """Register a paged (decode) layer's shape; every paged call then allocates the largest registered size."""
    global _PAGED_ROW_WORDS
    _PAGED_ROW_WORDS = max(_PAGED_ROW_WORDS, scratch_words_per_row(width, num_blocks, topk, source))


def set_decode_group(rows_per_request: int) -> None:
    """Consecutive decode rows that read the same pages: 1 + num_speculative_tokens on sm_120, which flattens a
    request's queries into rows. The paged scan then fetches a group's K once instead of once per row (it matters
    when a step's K exceeds the L2); any value is correct."""
    global _DECODE_GROUP
    _DECODE_GROUP = max(1, min(64, int(rows_per_request)))


def _topk(paged: bool, q, q_scale, kv, kv_scale, weights, lo, hi, block_table, width: int, topk: int, out,
          candidates, min_row_words: int) -> None:
    rows = q.shape[0]
    if rows == 0:
        return
    ext, dt = _ext(), _det_topk()
    max_rows = ext.max_rows()
    if rows > max_rows:  # the scan's per-CTA prefix sum bounds a launch's rows
        for r0 in range(0, rows, max_rows):
            sl = slice(r0, min(rows, r0 + max_rows))
            _topk(paged, q[sl], q_scale[sl], kv, kv_scale, weights[sl], lo[sl], hi[sl] if hi is not None else None,
                  block_table[sl] if block_table is not None else None, width, topk, out[sl],
                  candidates[sl] if candidates is not None else None, min_row_words)
        return
    source = candidates is not None
    num_blocks = candidates.shape[1] if source else topk
    wb = (width + BLOCK - 1) // BLOCK
    short_blocks, score_cols, logit_cols, row_words = _layout(width, num_blocks, topk, source)
    all_short = wb <= short_blocks
    sel_cols, col_cols = _up(num_blocks, 4), _up(topk, 4)
    buf = torch.empty(rows * max(row_words, min_row_words) + 16, dtype=torch.int32, device=q.device)
    o = 0

    def take(n_cols: int, dtype=torch.int32):
        nonlocal o
        t = buf[o:o + rows * n_cols].view(rows, n_cols)
        o += rows * n_cols
        return t.view(dtype) if dtype != torch.int32 else t

    scores = take(score_cols, torch.float32)
    logits = take(logit_cols, torch.float32)
    sel = take(sel_cols)[:, :num_blocks]
    cols = take(col_cols)
    r4 = _up(rows, 4)
    valid, sel_ends, end = buf[o:o + rows], buf[o + r4:o + r4 + rows], buf[o + 2 * r4:o + 2 * r4 + rows]

    if paged:
        ext.paged_scan(q, q_scale, kv, weights, lo, block_table, scores, valid, sel_ends, end, logits, num_blocks,
                       short_blocks, width, source, source, _DECODE_GROUP)
    else:
        ext.contiguous_scan(q, q_scale, kv, kv_scale, weights, lo, hi, scores, valid, sel_ends, end, logits,
                            num_blocks, short_blocks, width, source, source)
    kb = min(num_blocks, wb)
    if _STOP_AFTER == 1:
        return
    if all_short:
        # no row is long: the logits are the dense logits, column = token
        dt.topk_prefill(logits, None, end, out, rows, topk, True)
        if source:
            dt.topk_prefill(scores, None, sel_ends, sel, rows, kb, True)
            ext.finish(None, sel, None, topk, scores, candidates, kb, None, 0, rows)
        return
    dt.topk_prefill(scores, None, sel_ends, sel, rows, kb, True)
    if _STOP_AFTER == 2:
        return
    if paged:
        ext.paged_rescore(q, q_scale, kv, weights, lo, block_table, valid, sel, end, logits, short_blocks, width,
                          False)
    else:
        ext.contiguous_rescore(q, q_scale, kv, kv_scale, weights, lo, hi, valid, sel, end, logits, short_blocks,
                               width, False)
    if _STOP_AFTER == 3:
        return
    dt.topk_prefill(logits, None, end, cols, rows, topk, True)
    if _STOP_AFTER == 4:
        return
    ext.finish(cols, sel, out, topk, scores if source else None, candidates, kb, valid, short_blocks, rows)


def _candidates_topk(paged: bool, q, q_scale, kv, kv_scale, weights, lo, hi, block_table, width: int, topk: int, out,
                     candidates, min_row_words: int) -> None:
    rows = q.shape[0]
    if rows == 0:
        return
    ext, dt = _ext(), _det_topk()
    max_rows = ext.max_rows()
    if rows > max_rows:  # the rescore's per-CTA prefix sum bounds a launch's rows
        for r0 in range(0, rows, max_rows):
            sl = slice(r0, min(rows, r0 + max_rows))
            _candidates_topk(paged, q[sl], q_scale[sl], kv, kv_scale, weights[sl], lo[sl],
                             hi[sl] if hi is not None else None, block_table[sl] if block_table is not None else None,
                             width, topk, out[sl], candidates[sl], min_row_words)
        return
    num_blocks = candidates.shape[1]
    logit_cols, sel_cols, col_cols = num_blocks * BLOCK, _up(num_blocks, 4), _up(topk, 4)
    row_words = logit_cols + sel_cols + col_cols + 2
    buf = torch.empty(rows * max(row_words, min_row_words) + 8, dtype=torch.int32, device=q.device)
    o = rows * logit_cols
    logits = buf[:o].view(rows, logit_cols).view(torch.float32)
    sel = buf[o:o + rows * sel_cols].view(rows, sel_cols)[:, :num_blocks]
    o += rows * sel_cols
    cols = buf[o:o + rows * col_cols].view(rows, col_cols)
    o += rows * col_cols
    count = buf[o:o + rows]
    end = buf[o + _up(rows, 4):o + _up(rows, 4) + rows]
    ext.compact(candidates, lo, hi, sel, count, end, width, rows)
    if paged:
        ext.paged_rescore(q, q_scale, kv, weights, lo, block_table, count, sel, end, logits, num_blocks, width, True)
    else:
        ext.contiguous_rescore(q, q_scale, kv, kv_scale, weights, lo, hi, count, sel, end, logits, num_blocks, width,
                               True)
    dt.topk_prefill(logits, None, end, cols, rows, topk, True)
    ext.finish(cols, sel, out, topk, None, None, 0, None, 0, rows)


def _aligned(t: torch.Tensor, n: int) -> bool:
    return t.data_ptr() % n == 0


def paged_supported(kv_cache: torch.Tensor, block_table: torch.Tensor, q: torch.Tensor) -> bool:
    """The layouts the paged kernels take (a caller falls back to the dense path otherwise)."""
    page_kv = kv_cache.shape[1] if kv_cache.dim() == 4 else 0
    return (kv_cache.dim() == 4 and kv_cache.dtype == torch.uint8 and kv_cache.shape[2] == 1 and
            kv_cache.shape[3] == 68 and page_kv >= 16 and page_kv & (page_kv - 1) == 0 and
            kv_cache.stride(0) % 16 == 0 and _aligned(kv_cache, 16) and block_table.dim() == 2 and
            block_table.dtype == torch.int32 and block_table.stride(1) == 1 and _aligned(q, 4))


def contiguous_supported(kv: torch.Tensor, q: torch.Tensor) -> bool:
    return (kv.dim() == 2 and kv.shape[1] == 64 and kv.shape[0] > 0 and kv.is_contiguous() and _aligned(kv, 16) and
            _aligned(q, 4))


def _check_out(out: torch.Tensor, rows: int, topk: int, candidates) -> None:
    assert out.dtype == torch.int32 and out.dim() == 2 and out.shape[0] >= rows and out.shape[1] >= topk
    assert out.stride(1) == 1
    if candidates is not None:
        assert candidates.dtype == torch.int32 and candidates.dim() == 2 and candidates.shape[0] >= rows
        assert candidates.stride(1) == 1


def paged_topk(q: torch.Tensor, q_scale: torch.Tensor, kv_cache: torch.Tensor, weights: torch.Tensor,
               context_lens: torch.Tensor, block_table: torch.Tensor, width: int, topk: int, out: torch.Tensor,
               candidates: torch.Tensor | None = None) -> None:
    """Decode rows (one row per query). ``q`` int8 ``[rows, (1,) 32, 64]``, ``q_scale`` int32 ``[rows, (1,) 32]``,
    ``kv_cache`` uint8 ``[num_pages, page_kv, 1, 68]`` view, ``weights`` fp32 ``[rows, 32]``, ``context_lens`` int32
    ``[rows]``, ``block_table`` int32 ``[rows, P]``; ``width`` = the dense logits' width (max_model_len of the layer).
    With ``candidates`` the row is the candidate source's: its newest block pinned, the top ``candidates.shape[1]``
    blocks written there."""
    rows = q.shape[0]
    _check_out(out, rows, topk, candidates)
    lens = context_lens.reshape(-1)
    assert lens.dtype == torch.int32 and lens.numel() == rows
    _topk(True, q, q_scale, kv_cache, None, weights, lens if lens.is_contiguous() else lens.contiguous(), None,
          block_table, int(width), int(topk), out, candidates, _PAGED_ROW_WORDS)


def contiguous_topk(q: torch.Tensor, q_scale: torch.Tensor, kv: torch.Tensor, kv_scale: torch.Tensor,
                    weights: torch.Tensor, cu_ks: torch.Tensor, cu_ke: torch.Tensor, width: int, topk: int,
                    out: torch.Tensor, candidates: torch.Tensor | None = None) -> None:
    """Prefill-chunk rows over the packed K workspace: ``kv`` int8 ``[T, 64]``, ``kv_scale`` int32 ``[T]``, row r =
    workspace tokens ``[cu_ks[r], cu_ke[r])``; positions in ``out`` are relative to ``cu_ks[r]`` (the compressed
    columns of DeepGEMM's fp8_fp4_mqa_logits). ``width`` = the dense logits' width (the chunk's max_seqlen_k)."""
    rows = q.shape[0]
    _check_out(out, rows, topk, candidates)
    ks = cu_ks if cu_ks.is_contiguous() else cu_ks.contiguous()
    ke = cu_ke if cu_ke.is_contiguous() else cu_ke.contiguous()
    _topk(False, q, q_scale, kv, kv_scale.reshape(-1), weights, ks, ke, None, int(width), int(topk), out, candidates, 0)


def paged_candidates_topk(q: torch.Tensor, q_scale: torch.Tensor, kv_cache: torch.Tensor, weights: torch.Tensor,
                          context_lens: torch.Tensor, block_table: torch.Tensor, width: int, topk: int,
                          out: torch.Tensor, candidates: torch.Tensor) -> None:
    """A candidate consumer's decode rows: the top-k among the tokens of ``candidates[r]`` (the candidate source's
    blocks of the same row: ascending, -1 where dropped or unused)."""
    rows = q.shape[0]
    _check_out(out, rows, topk, candidates)
    lens = context_lens.reshape(-1)
    assert lens.dtype == torch.int32 and lens.numel() == rows
    _candidates_topk(True, q, q_scale, kv_cache, None, weights, lens if lens.is_contiguous() else lens.contiguous(),
                     None, block_table, int(width), int(topk), out, candidates, _PAGED_ROW_WORDS)


def contiguous_candidates_topk(q: torch.Tensor, q_scale: torch.Tensor, kv: torch.Tensor, kv_scale: torch.Tensor,
                               weights: torch.Tensor, cu_ks: torch.Tensor, cu_ke: torch.Tensor, width: int, topk: int,
                               out: torch.Tensor, candidates: torch.Tensor) -> None:
    """A candidate consumer's prefill-chunk rows (positions and candidate blocks relative to ``cu_ks[r]``)."""
    rows = q.shape[0]
    _check_out(out, rows, topk, candidates)
    ks = cu_ks if cu_ks.is_contiguous() else cu_ks.contiguous()
    ke = cu_ke if cu_ke.is_contiguous() else cu_ke.contiguous()
    _candidates_topk(False, q, q_scale, kv, kv_scale.reshape(-1), weights, ks, ke, None, int(width), int(topk), out,
                     candidates, 0)
