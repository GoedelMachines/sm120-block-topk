#!/usr/bin/env python3
"""The patched vLLM op (patch/patch_vllm_indexer_block_topk_sm120.py) with its switches on against the same op with them off.

Runs vLLM's own SparseAttnIndexer layer (the registered custom op torch.ops.vllm.sparse_attn_indexer) on a mixed batch
- flattened decode rows, then two prefill chunks - under a forward context with real indexer metadata dataclasses, for
the three layer kinds (ratio-2 plain, candidate source, candidate consumer), in three modes:
  dense    VLLM_MOET_INDEXER_BLOCK_TOPK=0                                      (the dense code, unchanged by the patch)
  decode   VLLM_MOET_INDEXER_BLOCK_TOPK=1                                      (decode rows block, prefill chunks dense:
                                                                               the default with the switch on)
  both     VLLM_MOET_INDEXER_BLOCK_TOPK=1 VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL=1
and compares the int32 top-k buffer and the candidate-block buffer of `decode` and `both` with `dense`, bit for bit. It
also checks which path ran for which part (call counts) and that the layer's construction registered the decode
scratch size.

Run with the patched file mounted over the image's (what the engine does):
    V=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
    EXTRA_DOCKER_ARGS="-v $PWD/patch/patched/model_executor/layers/sparse_attn_indexer.py:$V:ro -e VLLM_MOET_DET_TOPK=1
      -e VLLM_MOET_SPARSE_INDEXER_DIR=$PWD" ./run_in_image.sh python3 tests/test_vllm_block_patch.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NCAND, TOPK, num_sms, rand_q  # noqa: E402
from test_block_topk import make_cache  # noqa: E402

FAILS: list[str] = []
MML = 1 << 20


def build_batch(page_kv: int, kv3: torch.Tensor, gen, decode_ctx, prefill_reqs, chunk_split: int, flat: bool):
    """decode_ctx: context per decode row; prefill_reqs: [(request length, rows)] - a request's last `rows` positions
    are prefill rows; chunk_split: how many requests go to the first prefill chunk (the rest to a second); flat: the
    decode seq_lens 1-D [rows] as sm_120's flattened spec-decode builder writes them, else 2-D [rows, 1]."""
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata, DeepseekV32IndexerMetadata, DeepseekV32IndexerPrefillChunkMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )

    i32 = dict(dtype=torch.int32, device="cuda")
    pool = kv3.shape[0]
    D = len(decode_ctx)
    seq_lens = torch.tensor(decode_ctx, **i32).view(-1, 1)
    npages = max(1, (max(decode_ctx) + page_kv - 1) // page_kv)
    bt_d = torch.randint(0, pool, ((D + 7) // 8, npages), generator=gen, **i32).repeat_interleave(8, dim=0)[:D].contiguous()
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=bt_d, seq_lens=seq_lens.view(-1) if flat else seq_lens, decode_lens=torch.ones(D, **i32),
        requires_padding=False,
        schedule_metadata=get_paged_mqa_logits_metadata(seq_lens, page_kv, num_sms()))
    chunks = []
    token = D
    for reqs in (prefill_reqs[:chunk_split], prefill_reqs[chunk_split:]):
        if not reqs:
            continue
        cu = [0]
        for L, _ in reqs:
            cu.append(cu[-1] + L)
        ks, ke = [], []
        for i, (L, n) in enumerate(reqs):
            for pos in range(L - n, L):
                ks.append(cu[i])
                ke.append(cu[i] + pos + 1)
        rows = len(ks)
        max_pages = max((L + page_kv - 1) // page_kv for L, _ in reqs)
        chunks.append(DeepseekV32IndexerPrefillChunkMetadata(
            block_table=torch.randint(0, pool, (len(reqs), max_pages), generator=gen, **i32),
            cu_seqlen_ks=torch.tensor(ks, **i32), cu_seqlen_ke=torch.tensor(ke, **i32),
            cu_seq_lens=torch.tensor(cu, **i32), token_to_seq=torch.zeros(rows, **i32), total_seq_lens=cu[-1],
            max_seqlen_k=max(e - s for s, e in zip(ks, ke)), token_start=token, token_end=token + rows,
            num_reqs=len(reqs), local_cu_seq_lens=torch.tensor(cu, **i32), local_total_seq_lens=cu[-1],
            max_local_total_seq_lens=cu[-1]))
        token += rows
    N = token
    meta = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens.view(-1), max_seq_len=max(decode_ctx), slot_mapping=torch.zeros(N, dtype=torch.int64, device="cuda"),
        num_decodes=D, num_decode_tokens=D, num_prefills=len(prefill_reqs), num_prefill_tokens=N - D,
        decode=decode, prefill=DeepseekV32IndexerPrefillMetadata(chunks=chunks))
    return meta, N


def main():
    import vllm.model_executor.layers.sparse_attn_indexer as sai
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager

    assert hasattr(sai, "_moet_block_topk"), "the patched sparse_attn_indexer.py is not mounted over the image's"
    assert os.environ.get("VLLM_MOET_DET_TOPK") == "1" and os.environ.get("VLLM_MOET_CANDIDATE_DET_TOPK") == "1"
    torch.cuda.init()
    gen = torch.Generator(device="cuda").manual_seed(99)
    init_workspace_manager(torch.device("cuda:0"))

    def switch(on: bool, prefill: bool = False):
        os.environ["VLLM_MOET_INDEXER_BLOCK_TOPK"] = "1" if on else "0"
        os.environ["VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL"] = "1" if prefill else "0"
        sai._MOET_BLOCK_TOPK = None

    MODES = {"dense": (False, False), "decode": (True, False), "both": (True, True)}

    switch(True)
    bk = sai._moet_block_topk()
    assert bk is not None, "block_topk did not load"
    calls = {}
    for name in ("paged_topk", "contiguous_topk", "paged_candidates_topk", "contiguous_candidates_topk"):
        def wrap(fn, name=name):
            def counted(*a, **k):
                calls[name] = calls.get(name, 0) + 1
                return fn(*a, **k)
            return counted
        setattr(bk, name, wrap(getattr(bk, name)))

    # decode rows around the thresholds + long; prefill: two requests in chunk 1, one in chunk 2
    decode_ctx = [0, 3, 100, 511, 512, 513, 4096, 4097, 16384, 16385, 40000, 70000, 70001, 70002, 70003, 70004,
                  200000, 200001, 5000, 5001, 5002, 5003, 9, 300000]
    prefill_reqs = [(30003, 40), (700, 33), (45000, 51)]
    # seq_lens [rows, 1]: DeepGEMM's dense paged kernel (the reference here) asserts context_lens.dim() == 2, so that
    # is what the engine's flattened decode passes; block_topk itself takes either shape.
    for kind, page_kv, ratio, flat in (("plain", 64, 2, False), ("source", 128, 1, False), ("consumer", 128, 1, False)):
        kv3 = make_cache(1024, page_kv, gen)
        dctx = [c // ratio for c in decode_ctx]
        preqs = [(L // ratio, n) for L, n in prefill_reqs]
        meta, N = build_batch(page_kv, kv3, gen, dctx, preqs, 2, flat)
        prefix = f"model.layers.{kind}.indexer.k_cache"
        topk_buf = torch.empty(N + 8, TOPK, dtype=torch.int32, device="cuda")
        cand_buf = torch.empty(N + 8, NCAND, dtype=torch.int32, device="cuda") if kind != "plain" else None
        total_k = max(c.max_local_total_seq_lens for c in meta.prefill.chunks)
        reserved_before = bk._PAGED_ROW_WORDS
        with set_current_vllm_config(VllmConfig()):
            layers = {}
            for mode, (on, pre) in MODES.items():
                switch(on, pre)
                layers[mode] = sai.SparseAttnIndexer(
                    SimpleNamespace(prefix=prefix, kv_cache=kv3), 128, "ue8m0", TOPK, 128, MML // ratio, total_k + 64,
                    topk_buf, skip_k_cache_insert=True, use_fp4_cache=True, compress_ratio=ratio,
                    candidate_blocks=cand_buf, candidate_block_size=8 if cand_buf is not None else 0,
                    candidate_write=kind == "source")
        bk.set_decode_group(8)  # the layers registered 1 + VllmConfig().num_speculative_tokens = 1; serve as with 7
        want = 0 if kind == "consumer" else bk.scratch_words_per_row(MML // ratio, NCAND if kind == "source" else TOPK,
                                                                     TOPK, kind == "source")
        ok_reserve = bk._PAGED_ROW_WORDS >= max(reserved_before, want)
        q, qs, w = rand_q(N, gen)
        hidden = torch.zeros(N, 8, device="cuda")
        cand_src = None
        if kind == "consumer":
            # the candidates of these rows: the source layer's dense output for another q
            switch(False)
            q2, qs2, w2 = rand_q(N, gen)
            with set_current_vllm_config(VllmConfig()):
                src = sai.SparseAttnIndexer(
                    SimpleNamespace(prefix=prefix, kv_cache=kv3), 128, "ue8m0", TOPK, 128, MML, total_k + 64, topk_buf,
                    skip_k_cache_insert=True, use_fp4_cache=True, compress_ratio=1, candidate_blocks=cand_buf,
                    candidate_block_size=8, candidate_write=True)
            ctx = ForwardContext(no_compile_layers={}, attn_metadata={prefix: meta}, slot_mapping={})
            with override_forward_context(ctx):
                src.forward_cuda(hidden, (q2.view(torch.uint8), qs2), None, w2)
            cand_src = cand_buf.clone()
        results = {}
        for mode, (on, pre) in MODES.items():
            switch(on, pre)
            calls.clear()
            topk_buf.fill_(-5)
            if cand_buf is not None:
                cand_buf.copy_(cand_src) if cand_src is not None else cand_buf.fill_(-5)
            ctx = ForwardContext(no_compile_layers={}, attn_metadata={prefix: meta}, slot_mapping={})
            with override_forward_context(ctx):
                layers[mode].forward_cuda(hidden, (q.view(torch.uint8), qs), None, w)
            torch.cuda.synchronize()
            results[mode] = (topk_buf[:N].clone(), cand_buf[:N].clone() if cand_buf is not None else None, dict(calls))
        t0, c0, n0 = results["dense"]
        pfx = "paged_candidates_topk" if kind == "consumer" else "paged_topk"
        cfx = "contiguous_candidates_topk" if kind == "consumer" else "contiguous_topk"
        expect = {"dense": {}, "decode": {pfx: 1}, "both": {pfx: 1, cfx: 2}}
        ok = ok_reserve and n0 == expect["dense"]
        parts = []
        for mode in ("decode", "both"):
            t1, c1, n1 = results[mode]
            same_topk = bool((t0 == t1).all())
            same_cand = c0 is None or bool((c0 == c1).all())
            ran = n1 == expect[mode]
            ok = ok and same_topk and same_cand and ran
            parts.append(f"{mode}: top-k {'equal' if same_topk else 'DIFFER'}"
                         f"{'' if c0 is None else ', candidates ' + ('equal' if same_cand else 'DIFFER')}, block calls "
                         f"{n1}{'' if ran else ' (EXPECTED ' + str(expect[mode]) + ')'}")
            if not same_topk:
                r = int((t0 != t1).any(1).nonzero()[0])
                d = (t0[r] != t1[r]).nonzero().flatten()[:4].tolist()
                parts.append(f"row {r} cols {d}: dense {t0[r, d].tolist()} block {t1[r, d].tolist()}")
        print(f"{'ok' if ok else 'FAIL':4s} {kind:8s} op, {N} rows ({len(dctx)} decode + {N - len(dctx)} prefill in "
              f"{len(meta.prefill.chunks)} chunks) vs dense | " + " | ".join(parts) +
              f" | dense mode ran block calls {n0 or 'none'}; scratch registered {ok_reserve}", flush=True)
        if not ok:
            FAILS.append(kind)
    if FAILS:
        print(f"\nFAILED: {FAILS}")
        sys.exit(1)
    print("\nALL EXACT")


if __name__ == "__main__":
    main()
