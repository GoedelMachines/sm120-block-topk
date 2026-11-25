#!/usr/bin/env python3
"""Run DeepSeek-V4.1's sparse two-level indexer (attention_config.indexer_sparse_logits) on sm_120.

With ``indexer_sparse_logits`` the candidate-consuming indexer layers (24, 28, 32, 36 of V4.1-Flash) score only the
2048 candidate blocks (16,384 tokens) the candidate source (layer 20) published, instead of dense logits of shape
[rows, max_model_len]. vLLM implements it for SM100 only (DeepGEMM's sparse kernels, DeepSelect's bf16 top-k). This
patcher routes it on sm_120 to the sm120_sparse_indexer module (this repository):

  vllm/utils/deep_gemm.py                                 moet_sm120_sparse_indexer() loads the module; on sm_120
                                                          the four sparse impls point at it
  vllm/model_executor/layers/sparse_mqa_indexer.py        the SM100 gate admits sm_120 with the module; weights_dtype is
                                                          fp32 there (the dense path's weights, bit for bit)
  vllm/v1/attention/backends/mla/sparse_indexer.py        the builder admits sm_120 + the flattened decode layout
                                                          (sm_120 flattens spec-decode rows: one row per query, which is
                                                          what the sparse path wants); no DeepSelect layout check; the
                                                          prefill chunker budgets fp32 sparse logits
  vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py
                                                          sparse_topk_remap ranks fp32 sparse logits with det_topk_sm120
                                                          (the dense sm_120 path's top-k) instead of DeepSelect

Runtime switch: VLLM_MOET_SPARSE_INDEXER=1 (default off: nothing changes), module dir VLLM_MOET_SPARSE_INDEXER_DIR
(default /opt/vllm-moet/sm120_sparse_indexer, the directory holding the sm120_sparse_indexer package and build/).
It also needs det_topk_sm120 (VLLM_MOET_DET_TOPK_DIR, step 26) and the engine flag
--attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true, ...}'.

Anchored on the installed text (each anchor must match exactly once), idempotent. Usage:
    python3 patch_vllm_sparse_indexer_sm120.py /usr/local/lib/python3.12/dist-packages/vllm
"""

from __future__ import annotations

import pathlib
import sys

MARK = "[vllm-moet] sm_120 sparse indexer"


def patch(path: pathlib.Path, old: str, new: str, count: int = 1) -> None:
    s = path.read_text()
    n = s.count(old)
    assert n == count, f"anchor x{n} != {count} in {path}:\n{old}"
    path.write_text(s.replace(old, new))


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    dg = root / "utils/deep_gemm.py"
    layer = root / "model_executor/layers/sparse_mqa_indexer.py"
    builder = root / "v1/attention/backends/mla/sparse_indexer.py"
    kern = root / "model_executor/kernels/attention/dsa/sparse_mqa_logits.py"
    for p in (dg, layer, builder, kern):
        assert p.exists(), f"not a vLLM tree: {p}"
    if all(MARK in p.read_text() for p in (dg, layer, builder, kern)):
        print("already patched")
        return
    assert not any(MARK in p.read_text() for p in (dg, layer, builder, kern)), "partially patched tree"

    # ---- 1. deep_gemm.py: the loader, and the four impls on sm_120
    patch(
        dg,
        "def has_deep_gemm_sparse_mqa() -> bool:\n",
        f'''# {MARK} (sm120_sparse_indexer): DeepGEMM's sparse MQA-logits API on sm_120 with fp32 logits,
# bit-identical at the candidate tokens to the dense sm_120 kernels; VLLM_MOET_SPARSE_INDEXER=1 enables it.
_MOET_SM120_SPARSE = None  # None = not resolved, False = off / unavailable, else the module


def moet_sm120_sparse_indexer():
    """The sm120_sparse_indexer module when VLLM_MOET_SPARSE_INDEXER=1 on an sm_120 device and it loads, else None."""
    global _MOET_SM120_SPARSE
    if _MOET_SM120_SPARSE is None:
        _MOET_SM120_SPARSE = False
        if (
            os.environ.get("VLLM_MOET_SPARSE_INDEXER", "0") == "1"
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            import sys

            d = os.environ.get("VLLM_MOET_SPARSE_INDEXER_DIR", "/opt/vllm-moet/sm120_sparse_indexer")
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import sm120_sparse_indexer as _m

                _m._ext()
                _m._det_topk()
                _MOET_SM120_SPARSE = _m
                logger.info_once("vllm-moet: sm_120 sparse MQA logits from %s", _m.__file__)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vllm-moet: sm_120 sparse indexer unavailable: %r", exc)
    return _MOET_SM120_SPARSE or None


def has_deep_gemm_sparse_mqa() -> bool:
''',
    )
    patch(
        dg,
        """    _get_paged_sparse_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_sparse_mqa_logits_metadata", None
    )
""",
        f"""    _get_paged_sparse_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_sparse_mqa_logits_metadata", None
    )
    if (_moet_sp := moet_sm120_sparse_indexer()) is not None:  # {MARK}
        _fp8_fp4_sparse_mqa_logits_impl = _moet_sp.fp8_fp4_sparse_mqa_logits
        _fp8_fp4_paged_sparse_mqa_logits_impl = _moet_sp.fp8_fp4_paged_sparse_mqa_logits
        _get_sparse_mqa_logits_metadata_impl = _moet_sp.get_sparse_mqa_logits_metadata
        _get_paged_sparse_mqa_logits_metadata_impl = _moet_sp.get_paged_sparse_mqa_logits_metadata
""",
    )

    # ---- 2. the layer: gate and weights dtype
    patch(
        layer,
        "from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa\n",
        "from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa\n"
        f"from vllm.utils.deep_gemm import moet_sm120_sparse_indexer  # {MARK}\n",
    )
    patch(
        layer,
        '''    weights_dtype = torch.bfloat16
    """Per-head weights dtype the sparse kernels take. The fused Q RoPE-quant
    kernel writes it directly so no cast runs per step."""
''',
        f'''    class _WeightsDtype:  # {MARK}: fp32 there, the dense sm_120 path's weights
        def __get__(self, obj, owner) -> torch.dtype:
            return torch.float32 if moet_sm120_sparse_indexer() is not None else torch.bfloat16

    weights_dtype = _WeightsDtype()
    """Per-head weights dtype the sparse kernels take. The fused Q RoPE-quant
    kernel writes it directly so no cast runs per step."""
''',
    )
    patch(
        layer,
        """        super().__init__()
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_deep_gemm_sparse_mqa()
        ):
            raise ValueError(
                "SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8."
            )
        if not has_deep_select():
""",
        f"""        super().__init__()
        _moet_sm120 = moet_sm120_sparse_indexer() is not None  # {MARK}
        if not _moet_sm120 and not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_deep_gemm_sparse_mqa()
        ):
            raise ValueError(
                "SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8."
            )
        if not _moet_sm120 and not has_deep_select():
""",
    )

    # ---- 3. the metadata builder
    patch(
        builder,
        "from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa\n",
        "from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa\n"
        f"from vllm.utils.deep_gemm import moet_sm120_sparse_indexer  # {MARK}\n",
    )
    patch(
        builder,
        """        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError(f"{prefix} requires an SM100-class GPU.")
""",
        f"""        # {MARK}: the sm120_sparse_indexer module stands in for DeepGEMM's
        # SM100 kernels and det_topk_sm120 for DeepSelect; sm_120 flattens decode rows.
        moet_sm120 = moet_sm120_sparse_indexer() is not None
        if not moet_sm120 and not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError(f"{{prefix}} requires an SM100-class GPU.")
""",
    )
    patch(
        builder,
        """        if not self.supports_varlen:
            raise ValueError(
""",
        """        # sm_120: one row per query = the flattened spec-decode layout, or plain decode (next_n == 1)
        if not (
            self.supports_varlen
            or (moet_sm120 and (self.use_flattening or self.num_speculative_tokens == 0))
        ):
            raise ValueError(
""",
    )
    patch(
        builder,
        """        if not has_deep_select():
            raise ValueError(
""",
        """        if not moet_sm120 and not has_deep_select():
            raise ValueError(
""",
    )
    patch(
        builder,
        """        if topk_backend not in ("auto", "deep_select"):
""",
        """        if not moet_sm120 and topk_backend not in ("auto", "deep_select"):
""",
    )
    patch(
        builder,
        """        check_deep_select_layout(num_sparse_cols, topk_tokens)
        # Sparse logits are bf16 [rows, num_sparse_cols]; make the prefill
        # chunker (which budgets fp32 [rows, seq_len]) size chunks as if every
        # row were this wide.
        self.min_split_seq_len = (num_sparse_cols * 2 + 3) // 4
""",
        """        if moet_sm120:
            # sm_120: fp32 sparse logits [rows, num_sparse_cols], ranked by det_topk_sm120.
            self.min_split_seq_len = num_sparse_cols
        else:
            check_deep_select_layout(num_sparse_cols, topk_tokens)
            # Sparse logits are bf16 [rows, num_sparse_cols]; make the prefill
            # chunker (which budgets fp32 [rows, seq_len]) size chunks as if every
            # row were this wide.
            self.min_split_seq_len = (num_sparse_cols * 2 + 3) // 4
""",
    )

    # ---- 4. the kernels module: fp32 weights admitted (decode and prefill entry points), the top-k on fp32 logits
    patch(
        kern,
        """    assert weights.dtype == torch.bfloat16, weights.dtype
""",
        f"""    assert weights.dtype in (torch.bfloat16, torch.float32), weights.dtype  # {MARK}: fp32
""",
        count=2,
    )
    patch(
        kern,
        """    rows, width = logits.shape
    assert logits.dtype == torch.bfloat16 and width >= topk_tokens
    assert end.dtype == torch.int32 and end.shape == (rows,)
    cols = col_indices[:rows]
    deep_select_topk(logits, topk_tokens, end=end, output_idx=cols)
""",
        f"""    rows, width = logits.shape
    assert end.dtype == torch.int32 and end.shape == (rows,)
    cols = col_indices[:rows]
    if logits.dtype == torch.float32:
        # {MARK}: fp32 sparse logits, ranked by the dense sm_120
        # path's deterministic top-k (equal values to the lower column = the lower token).
        from vllm.utils.deep_gemm import moet_sm120_sparse_indexer

        assert width >= topk_tokens
        moet_sm120_sparse_indexer().sparse_topk(logits, end, topk_tokens, cols)
    else:
        assert logits.dtype == torch.bfloat16 and width >= topk_tokens
        deep_select_topk(logits, topk_tokens, end=end, output_idx=cols)
""",
    )
    print(f"patched: {dg.name}, {layer.name}, {builder.name}, {kern.name}")


if __name__ == "__main__":
    main()
