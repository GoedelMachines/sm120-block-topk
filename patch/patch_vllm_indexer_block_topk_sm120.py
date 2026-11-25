#!/usr/bin/env python3
"""Run DeepSeek-V4.1's indexer layers on sm_120 without full-width logits (sm120_sparse_indexer.block_topk).

Today every indexer layer of the dense path (SparseAttnIndexer) materialises fp32 logits [rows, width]: width =
max_model_len / compress_ratio in decode (4 MiB per row at 1M for the ratio-1 layers, 2 MiB for the ratio-2 layers),
which a decode CUDA graph holds for its largest batch. With VLLM_MOET_INDEXER_BLOCK_TOPK=1 the op computes the same
int32 top-k rows and the same candidate blocks from fp32 block maxima [rows, width / 8] and the logits of the selected
blocks [rows, 16384] (0.57 MiB per row at 1M, one block of the graph pool for all eight layers):

  layers 2, 8, 14 (ratio 2), layer 20 (the candidate source)   block_topk.paged_topk / contiguous_topk
  layers 24, 28, 32, 36 (candidate consumers)                  block_topk.paged_candidates_topk / contiguous_candidates_topk

Switches:
  VLLM_MOET_INDEXER_BLOCK_TOPK=1          the decode branch (the rows CUDA graphs capture: where the memory is)
  VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL=1  the prefill-chunk branch as well (default 0: prefill chunks stay on the dense
                                          code - they are eager, hold no graph memory, and the engine measured the
                                          block path 12-26 % slower on fresh prefill)
A step with decode rows and prefill chunks routes each part by its own switch; both parts write the dense path's bits
(the candidate blocks are identical either way), so every mix is exact.

One file is patched, vllm/model_executor/layers/sparse_attn_indexer.py:
  - _moet_block_topk()        loads the module when VLLM_MOET_INDEXER_BLOCK_TOPK=1 on sm_120 with VLLM_MOET_DET_TOPK=1
                              and VLLM_MOET_CANDIDATE_DET_TOPK=1 (it reproduces those selections);
                              _moet_block_topk(prefill=True) also wants VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL=1
  - SparseAttnIndexer.__init__ registers the layer's decode shape (one scratch size for every decode call) and the
                              rows per request (1 + num_speculative_tokens: the scan fetches their shared K once)
  - sparse_attn_indexer        the decode branch (and, with the prefill switch, the prefill-chunk branch) calls the
                              module instead of fp8_fp4_(paged_)mqa_logits + select_candidate_blocks /
                              apply_candidate_mask + top-k when the layout is the one sm_120 serves (MXFP4 cache, fp32
                              weights, one row per query, 8-token candidate blocks, no DCP/PCP); anything else stays
                              on the dense code below it.

Module dir: VLLM_MOET_SPARSE_INDEXER_DIR (default /opt/vllm-moet/sm120_sparse_indexer: the directory holding the
sm120_sparse_indexer package and build/). Independent of attention_config.indexer_sparse_logits (leave it off: the
consumers are covered here, with the dense backend's KV layout).

Anchored on the installed text (each anchor must match exactly once), idempotent. Usage:
    python3 patch_vllm_indexer_block_topk_sm120.py /usr/local/lib/python3.12/dist-packages/vllm
"""

from __future__ import annotations

import pathlib
import sys

MARK = "[vllm-moet] indexer block top-k"


def patch(path: pathlib.Path, old: str, new: str) -> None:
    s = path.read_text()
    n = s.count(old)
    assert n == 1, f"anchor x{n} != 1 in {path}:\n{old}"
    path.write_text(s.replace(old, new))


LOADER = f'''# {MARK} (sm120_sparse_indexer.block_topk): the same top-k rows and candidate blocks as the dense
# path below from block maxima [rows, width / 8] + the selected blocks' logits, instead of fp32 logits [rows, width].
# VLLM_MOET_INDEXER_BLOCK_TOPK=1 enables it for decode rows (where CUDA graphs hold the logits);
# VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL=1 for prefill chunks too (default off: eager, no graph memory, and slower
# in the engine). It reproduces det_topk's selections, so it needs steps 26 and 31 on.
_MOET_BLOCK_TOPK = None  # None = not resolved, False = off / unavailable, else the module
_MOET_BLOCK_TOPK_PREFILL = False  # resolved with it


def _moet_block_topk(prefill: bool = False):
    """The module for a decode call, or (prefill=True) for a prefill chunk; None = the dense path."""
    global _MOET_BLOCK_TOPK, _MOET_BLOCK_TOPK_PREFILL
    if _MOET_BLOCK_TOPK is None:
        import os
        import sys

        _MOET_BLOCK_TOPK = False
        _MOET_BLOCK_TOPK_PREFILL = (
            os.environ.get("VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL", "0") == "1"
        )
        if (
            os.environ.get("VLLM_MOET_INDEXER_BLOCK_TOPK", "0") == "1"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 12
        ):
            if (
                os.environ.get("VLLM_MOET_DET_TOPK", "0") != "1"
                or os.environ.get("VLLM_MOET_CANDIDATE_DET_TOPK", "0") != "1"
            ):
                logger.warning(
                    "vllm-moet: VLLM_MOET_INDEXER_BLOCK_TOPK needs VLLM_MOET_DET_TOPK=1 and "
                    "VLLM_MOET_CANDIDATE_DET_TOPK=1; the indexer stays on the dense path"
                )
                return None
            d = os.environ.get("VLLM_MOET_SPARSE_INDEXER_DIR", "/opt/vllm-moet/sm120_sparse_indexer")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                from sm120_sparse_indexer import block_topk as _m

                _m._ext()
                _m._det_topk()
                _MOET_BLOCK_TOPK = _m
                logger.info_once(
                    "vllm-moet: indexer block top-k from %s (decode rows%s)",
                    _m.__file__,
                    " and prefill chunks"
                    if _MOET_BLOCK_TOPK_PREFILL
                    else "; prefill chunks dense",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vllm-moet: indexer block top-k unavailable, dense path: %r", exc)
    if prefill and not _MOET_BLOCK_TOPK_PREFILL:
        return None
    return _MOET_BLOCK_TOPK or None


'''

PREFILL = f'''                _moet_bt = _moet_block_topk(prefill=True)  # {MARK}
                if (
                    _moet_bt is not None
                    and use_fp4_cache
                    and q_scale_slice is not None
                    and dcp_world_size == 1
                    and deinterleave_idx is None
                    and (candidate_blocks is None or candidate_block_size == 8)
                    and weights.dtype == torch.float32
                    and topk_indices.dtype == torch.int32
                    and topk_indices.stride(-1) == 1
                    and cu_seqlen_ks.dtype == torch.int32
                    and _moet_bt.contiguous_supported(k_quant_cast, q_slice_cast)
                ):
                    # The dense path's top-k rows (and candidate blocks) without
                    # its [rows, max_seqlen_k] logits; dcp_world_size == 1, so
                    # there is no merge after it.
                    if candidate_blocks is None or candidate_write:
                        _moet_bt.contiguous_topk(
                            q_slice_cast,
                            q_scale_slice,
                            k_quant_cast,
                            k_scale_cast,
                            weights[row_start:row_end],
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk.max_seqlen_k,
                            topk_tokens,
                            topk_indices,
                            candidates=(
                                candidate_blocks[chunk.token_start : chunk.token_end]
                                if candidate_write
                                else None
                            ),
                        )
                    else:
                        _moet_bt.contiguous_candidates_topk(
                            q_slice_cast,
                            q_scale_slice,
                            k_quant_cast,
                            k_scale_cast,
                            weights[row_start:row_end],
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk.max_seqlen_k,
                            topk_tokens,
                            topk_indices,
                            candidate_blocks[chunk.token_start : chunk.token_end],
                        )
                    continue
'''

DECODE = f'''        _moet_bt = _moet_block_topk()  # {MARK}
        _moet_bt_ok = (
            _moet_bt is not None
            and use_fp4_cache
            and padded_q_scale is not None
            and next_n == 1
            and not needs_padded_path
            and dcp_world_size == 1
            and decode_metadata.global_seq_lens is None
            and (candidate_blocks is None or candidate_block_size == 8)
            and seq_lens.dtype == torch.int32
            and seq_lens.numel() == batch_size
            and decode_metadata.block_table.shape[0] >= batch_size
            and weights.dtype == torch.float32
            and topk_indices_buffer.dtype == torch.int32
            and _moet_bt.paged_supported(
                kv_cache, decode_metadata.block_table, padded_q_quant_cast
            )
        )
        if _moet_bt is not None and not _moet_bt_ok and num_decode_tokens > 0:
            # Said once: this decode layout keeps its full-width logits (and
            # their CUDA graph memory).
            logger.info_once(
                "vllm-moet: indexer block top-k does not take this decode layout "
                "(next_n=%d padded=%s seq_lens=%s/%s block_table=%s weights=%s "
                "candidate_block_size=%d dcp=%d); dense path",
                next_n,
                needs_padded_path,
                tuple(seq_lens.shape),
                seq_lens.dtype,
                tuple(decode_metadata.block_table.shape),
                weights.dtype,
                candidate_block_size,
                dcp_world_size,
            )
        if _moet_bt_ok and num_decode_tokens > 0:
            # One row per query (sm_120 flattens spec-decode rows): the dense
            # path's top-k rows (and candidate blocks) without its
            # [rows, max_model_len] logits. Nothing follows the decode branch.
            topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
            if candidate_blocks is None or candidate_write:
                _moet_bt.paged_topk(
                    padded_q_quant_cast,
                    padded_q_scale,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    max_model_len,
                    topk_tokens,
                    topk_indices,
                    candidates=(
                        candidate_blocks[:num_padded_tokens] if candidate_write else None
                    ),
                )
            else:
                _moet_bt.paged_candidates_topk(
                    padded_q_quant_cast,
                    padded_q_scale,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    max_model_len,
                    topk_tokens,
                    topk_indices,
                    candidate_blocks[:num_padded_tokens],
                )
            return topk_indices_buffer
'''

INIT = f'''        # {MARK}: every decode call asks for the largest layer's scratch
        # size, so a captured step reuses one block of the graph pool; a request's
        # 1 + num_speculative_tokens flattened decode rows read the same pages.
        _moet_bt = _moet_block_topk() if use_fp4_cache else None
        if _moet_bt is not None:
            _moet_bt.set_decode_group(
                1 + get_current_vllm_config().num_speculative_tokens
            )
            if candidate_blocks is None or candidate_write:
                _moet_bt.reserve_paged(
                    max_model_len,
                    candidate_blocks.shape[1]
                    if candidate_blocks is not None
                    else topk_tokens,
                    topk_tokens,
                    candidate_write,
                )
'''


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    f = root / "model_executor/layers/sparse_attn_indexer.py"
    assert f.exists(), f"not a vLLM tree: {f}"
    if MARK in f.read_text():
        print("already patched")
        return

    # ---- 1. the loader, after step 26's
    patch(
        f,
        "# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.\n",
        LOADER + "# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.\n",
    )
    # ---- 2. prefill chunk: before the dense logits
    anchor = '''                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
'''
    patch(f, anchor, PREFILL + anchor)
    # ---- 3. decode: before the dense logits
    anchor = '''        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
'''
    patch(f, anchor, DECODE + anchor)
    # ---- 4. the layer registers its decode shape
    anchor = '''        self.candidate_write = candidate_write
        self.dense_mha_metadata_layer_name = ""
'''
    patch(f, anchor, anchor + INIT)
    print(f"patched: {f}")


if __name__ == "__main__":
    main()
