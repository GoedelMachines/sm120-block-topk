"""DeepGEMM's sparse MQA-logits API on sm_120, exact against the dense sm_120 path.

DeepSeek-V4.1's candidate-consuming indexer layers (``attention_config.indexer_sparse_logits``) score only the
candidate blocks the candidate-source layer published. vLLM calls four DeepGEMM functions for that
(``vllm/utils/deep_gemm.py``); DeepGEMM implements them for SM100 only. This module provides the same four for sm_120
with one deliberate difference: the logits are **fp32**, bit-identical at every candidate token to what DeepGEMM's sm_120
dense kernels (``fp8_fp4_paged_mqa_logits`` / ``fp8_fp4_mqa_logits``, which vLLM ranks today) compute there, given the
same fp32 weights. bf16 logits (the SM100 contract) would round near-ties into the top-k; fp32 keeps the selection
identical to the dense path's.

    get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
                                         qk_dtype, sparse_block_kv) -> int32 [rows, S + 2]
    fp8_fp4_paged_sparse_mqa_logits((q, q_scale), kv_cache, weights, metadata, S, sparse_block_kv)
                                         -> fp32 [rows, S * sparse_block_kv]
    get_sparse_mqa_logits_metadata(cu_seqlen_ks, cu_seqlen_ke, num_kv_tokens, sparse_kv_block_indices, qk_dtype,
                                   sparse_block_kv, use_unaligned_ks=False) -> int32 [rows, S + 2]
    fp8_fp4_sparse_mqa_logits((q, q_scale), (kv, kv_scale), weights, metadata, S, sparse_block_kv,
                              use_unaligned_ks=False) -> fp32 [rows, S * sparse_block_kv]
    sparse_topk(logits, end, topk, out)   # the dense path's deterministic top-k (det_topk_sm120) on the sparse columns

Column ``j * sbk + o`` scores token ``o`` of sparse block ``sparse_kv_block_indices[row, j]`` (paged: context position
``block * sbk + o``; contiguous: workspace token ``block * sbk + (ks % sbk if use_unaligned_ks else 0) + o``). Columns
whose token lies outside the row's range (paged ``[0, context_len)``, contiguous ``[ks, ke)``), and blocks that repeat
the previous one (DeepGEMM's padding convention), read ``-inf``.

Only MXFP4 Q/K (packed e2m1 viewed as int8, packed UE8M0 int32 scales, 32 heads x 128 dims) is implemented: the
indexer cache ``indexer_sparse_logits`` requires. ``weights`` must be fp32 for exactness (bf16 is accepted and upcast).

The CUDA extension is imported prebuilt from ``$VLLM_MOET_SPARSE_INDEXER_BUILD`` (default ``<package>/../build``) when
present, else compiled on first use with torch.utils.cpp_extension (~40 s; ``$TORCH_EXTENSIONS_DIR``).
"""

from __future__ import annotations

import functools
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT_NAME = "vllm_moet_sparse_mqa_logits_sm120"
_SRC = _HERE / "csrc" / "sparse_mqa_logits_sm120.cu"
_NVCC_FLAGS = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a"]

__all__ = [
    "get_paged_sparse_mqa_logits_metadata",
    "fp8_fp4_paged_sparse_mqa_logits",
    "get_sparse_mqa_logits_metadata",
    "fp8_fp4_sparse_mqa_logits",
    "sparse_topk",
    "available",
]


def _build_dir() -> Path:
    return Path(os.environ.get("VLLM_MOET_SPARSE_INDEXER_BUILD", str(_HERE.parent / "build")))


def _compile(name: str, src: Path, build_directory: str | None, verbose: bool):
    from torch.utils.cpp_extension import load

    # the -gencode below is the only target (block-scaled MMA needs sm_120a); keep torch from adding plain sm_120
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    try:
        kwargs = {} if build_directory is None else {"build_directory": build_directory}
        return load(name=name, sources=[str(src)], extra_cuda_cflags=_NVCC_FLAGS, extra_cflags=["-O3", "-std=c++20"],
                    verbose=verbose, **kwargs)
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def _load_ext(name: str, src: Path, subdir: str = ""):
    """The prebuilt extension from the build dir when it is not older than its source, else compiled on first use."""
    so = _build_dir() / subdir / f"{name}.so"
    if so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        loader = importlib.machinery.ExtensionFileLoader(name, str(so))
        spec = importlib.util.spec_from_file_location(name, str(so), loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod
    return _compile(name, src, None, False)


def _build_ext(name: str, src: Path, subdir: str, out_dir: str | os.PathLike | None) -> Path:
    out = (Path(out_dir) if out_dir is not None else _build_dir()) / subdir
    out.mkdir(parents=True, exist_ok=True)
    _compile(name, src, str(out), True)
    return out / f"{name}.so"


@functools.lru_cache(maxsize=None)
def _ext():
    return _load_ext(_EXT_NAME, _SRC)


def build(out_dir: str | os.PathLike | None = None) -> list[Path]:
    """Compile both extensions into ``out_dir`` (default: the build dir) and return the .so paths."""
    from . import block_topk

    return [_build_ext(_EXT_NAME, _SRC, "", out_dir),
            _build_ext(block_topk.EXT_NAME, block_topk.SRC, block_topk.EXT_NAME, out_dir)]


@functools.lru_cache(maxsize=None)
def _det_topk():
    d = os.environ.get("VLLM_MOET_DET_TOPK_DIR", "/opt/vllm-moet/dsv41_sm120/det_topk")
    if d not in sys.path:
        sys.path.insert(0, d)
    import det_topk_sm120

    det_topk_sm120._ext()
    return det_topk_sm120


def available() -> bool:
    """sm_120 device, the extension loads, and the deterministic top-k loads."""
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            return False
        _ext()
        _det_topk()
        return True
    except Exception:  # noqa: BLE001
        return False


def _check_mxfp4(qk_dtype) -> None:
    if qk_dtype not in (torch.int8, torch.uint8):
        raise NotImplementedError(f"sm_120 sparse MQA logits: MXFP4 (int8-packed) Q/K only, got {qk_dtype}")


def _i32(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32, t.dtype
    return t if t.is_contiguous() else t.contiguous()


# Pages are not known to the metadata call; the logits call bounds them by the cache it gets.
_MAX_PACKED_BLOCKS = 1 << 26


def get_paged_sparse_mqa_logits_metadata(context_lens: torch.Tensor, block_table: torch.Tensor,
                                         indices: torch.Tensor | None, page_kv: int,
                                         sparse_kv_block_indices: torch.Tensor, qk_dtype: torch.dtype,
                                         sparse_block_kv: int) -> torch.Tensor:
    """``indices`` (row -> request) only pairs rows for SM100's scheduler; ``block_table`` is already per row."""
    _check_mxfp4(qk_dtype)
    rows = sparse_kv_block_indices.shape[0]
    lens = _i32(context_lens.reshape(-1))
    assert lens.numel() == rows and block_table.shape[0] == rows and block_table.stride(-1) == 1
    num_pages_bound = _MAX_PACKED_BLOCKS // (page_kv // sparse_block_kv) - 1
    return _ext().paged_metadata(lens, block_table, sparse_kv_block_indices, int(page_kv), int(sparse_block_kv),
                                 num_pages_bound)


def fp8_fp4_paged_sparse_mqa_logits(q: tuple[torch.Tensor, torch.Tensor], kv_cache: torch.Tensor,
                                    weights: torch.Tensor, metadata: torch.Tensor, num_max_sparse_blocks: int,
                                    sparse_block_kv: int) -> torch.Tensor:
    q_values, q_scale = q
    assert q_scale is not None, "MXFP4 Q needs its packed UE8M0 scales"
    return _ext().paged_sparse_logits(q_values, q_scale, kv_cache, weights, metadata, int(num_max_sparse_blocks),
                                      int(sparse_block_kv))


def get_sparse_mqa_logits_metadata(cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, num_kv_tokens: int,
                                   sparse_kv_block_indices: torch.Tensor, qk_dtype: torch.dtype,
                                   sparse_block_kv: int, use_unaligned_ks: bool = False) -> torch.Tensor:
    _check_mxfp4(qk_dtype)
    return _ext().contiguous_metadata(_i32(cu_seqlen_ks), _i32(cu_seqlen_ke), int(num_kv_tokens),
                                      sparse_kv_block_indices, int(sparse_block_kv), bool(use_unaligned_ks))


def fp8_fp4_sparse_mqa_logits(q: tuple[torch.Tensor, torch.Tensor], kv: tuple[torch.Tensor, torch.Tensor],
                              weights: torch.Tensor, metadata: torch.Tensor, num_max_sparse_blocks: int,
                              sparse_block_kv: int, use_unaligned_ks: bool = False) -> torch.Tensor:
    """``use_unaligned_ks`` is fixed by the metadata (it placed the blocks); accepted for API parity."""
    q_values, q_scale = q
    kv_values, kv_scale = kv
    assert q_scale is not None, "MXFP4 Q needs its packed UE8M0 scales"
    return _ext().contiguous_sparse_logits(q_values, q_scale, kv_values, kv_scale.reshape(-1), weights, metadata,
                                           int(num_max_sparse_blocks), int(sparse_block_kv))


def sparse_topk(logits: torch.Tensor, end: torch.Tensor, topk: int, out: torch.Tensor) -> torch.Tensor:
    """The k largest of ``logits[r, :end[r]]`` (equal values to the lower column, NaN above +inf), as ascending
    columns then -1 — det_topk_sm120, the same kernel the dense sm_120 indexer path ranks with. Since the sparse
    columns are in ascending token order, the selection equals the dense path's at the same tokens."""
    assert logits.dtype == torch.float32 and end.dtype == torch.int32
    rows = logits.shape[0]
    _det_topk().topk_prefill(logits, None, end if end.is_contiguous() else end.contiguous(), out, rows, topk, True)
    return out
