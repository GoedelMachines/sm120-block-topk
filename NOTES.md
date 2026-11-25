# sm_120 indexer without full-width logits (DeepSeek-V4.1-Flash)

Two things live here:

- **Block top-k** (section 2, `sm120_sparse_indexer/block_topk.py`, switch `VLLM_MOET_INDEXER_BLOCK_TOPK=1`): all eight
  indexer layers return today's top-k rows and candidate blocks from block maxima + the selected blocks' logits.
  Decode CUDA-graph memory of the indexer sequence: 6.000 -> 0.570 MiB per row. **This is the one to serve.**
- **Sparse MQA-logits API** (section 1, `sm120_sparse_indexer/__init__.py`, `attention_config.indexer_sparse_logits` +
  `VLLM_MOET_SPARSE_INDEXER=1`): DeepGEMM's four sparse functions for sm_120, for the candidate consumers only. Alone
  it saves no graph memory (layer 20 and the ratio-2 layers keep the peak); block top-k does not need it.

# 1. Sparse MQA-logits API (`indexer_sparse_logits=True`)

What it is: DeepGEMM's sparse MQA-logits API (four functions) for sm_120, plus a four-file vLLM patch that routes
`attention_config.indexer_sparse_logits` to it. The candidate-consumer indexer layers then score only the 2,048
candidate blocks (16,384 tokens) per row instead of materialising `[rows, max_model_len]` fp32 logits.

## 1.1 Spec — the four functions as vLLM calls them

V4.1-Flash geometry: `index_n_heads` H = 32, `index_head_dim` D = 128, MXFP4 indexer cache (64 B e2m1 pairs + 4 B
UE8M0 per key = 68 B), `index_topk` = 512, `candidate_block_size` = 8, `candidate_topk_blocks` = 2048,
`candidate_source_layer_id` = 20 (writes candidates, stays dense), consumers = indexer layers 24, 28, 32, 36 (ratio 1,
128-row pages). `pick_sparse_block_kv(8)` = 8, so S = num_max_sparse_blocks = 2048, width = S * 8 = 16,384 columns.

Callers: `vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py` (`sparse_mqa_logits_paged_decode`,
`sparse_mqa_logits_prefill_chunk`) via the wrappers in `vllm/utils/deep_gemm.py`.

Shared input: `sparse_kv_block_indices` si, int32 `[rows, S]`, produced by `candidate_blocks_to_sparse_indices`
(Triton) from the candidate blocks: each row's valid blocks (inside `[0, ke - ks)`) sorted ascending (unique in
production: the candidate source's deterministic top-k), the tail padded by repeating the last valid block (or `ks//sbk`
if none); plus `end[rows]` = number of valid sparse columns (a prefix; only the last valid block can be partial).
Not passed to the logits functions: they get only the metadata.

| function | arguments (as called) | returns |
|---|---|---|
| `get_paged_sparse_mqa_logits_metadata` | `context_lens` int32 `[rows]` (decode `seq_lens.view(-1)`, per-row compressed context), `block_table` int32 `[rows, P]` (per row; stride(-1)=1), `indices` int32 `[rows]` (row -> request; only pairs rows for SM100's scheduler), `page_kv` = `kv_cache.shape[1]` (128), si (context-relative block ids), `qk_dtype` = int8 (MXFP4), `sparse_block_kv` = 8 | opaque schedule tensor |
| `fp8_fp4_paged_sparse_mqa_logits` | `q = (int8 [rows, 1, 32, 64], int32 [rows, 1, 32])`, `kv_cache` uint8 `[num_pages, page_kv, 1, 68]` view (a page = `page_kv x 64 B` values then `page_kv x 4 B` scales; page stride 256-B aligned), `weights [rows, 32]`, metadata, S, sbk | `[rows, S*sbk]`; column `j*sbk+o` = token `si[r,j]*sbk + o` |
| `get_sparse_mqa_logits_metadata` | `cu_seqlen_ks`, `cu_seqlen_ke` int32 `[rows]` (bounds in the packed K-gather workspace), `num_kv_tokens` = workspace rows, si (= request-local candidate + `ks//sbk`), int8, sbk, `use_unaligned_ks=True` | opaque schedule tensor |
| `fp8_fp4_sparse_mqa_logits` | `q = (int8 [rows, 32, 64], int32 [rows, 32])`, `kv = (int8 [T, 64], int32 [T])` (packed workspace from `cp_gather_indexer_k_quant_cache`), `weights [rows, 32]`, metadata, S, sbk, `use_unaligned_ks=True` | `[rows, S*sbk]`; column `j*sbk+o` = workspace token `si[r,j]*sbk + ks%sbk + o` |

Value at a column: `sum_h relu(<q_h, k>) * w_h` with block-scaled MXFP4 dot products (scales: 4 UE8M0 bytes per row,
one per 32 elements), the Q scales separate (not folded into the weights).

Then vLLM ranks: `sparse_topk_remap` = top-512 of each row's first `end` columns (SM100: DeepSelect on bf16), and the
Triton remap turns column `c` into request-local position `(si[r, c//sbk] - ks//sbk)*sbk + c%sbk` (-1 outside).

SM100 contract vs this implementation:

| | DeepGEMM SM100 | sm120_sparse_indexer |
|---|---|---|
| logits dtype | bf16 | **fp32** |
| weights | bf16 | **fp32** (bf16 accepted, upcast — not exact then) |
| metadata | DeepGEMM schedule | int32 `[rows, S+2]` (below) |
| top-k | DeepSelect (bf16, SM100 only) | `det_topk_sm120` (the dense sm_120 path's top-k) |
| repeated blocks | scored again ("tolerated but wasteful") | scored once; repeats read `-inf` |
| tokens outside the row's range | — | `-inf` |

Why fp32: the dense sm_120 path vLLM runs today ranks fp32 logits (DeepGEMM's sm_120 kernels return fp32 and the
weights are fp32). Equal top-k needs the same values; bf16 would round distinct logits into ties.

Metadata (int32 `[rows, S + 2]`, built on the device, graph-safe):
- paged: `entry[r, j] = (physical sparse block << 5) | n_valid`, physical sparse block = `page * (page_kv/sbk) +
  (pos % page_kv)/sbk` with `page = block_table[r, pos // page_kv]`, `pos = si*sbk`, `n_valid = min(sbk, ctx - pos)`;
  `-1` = skip (out of range, repeat of the previous block, negative id, page id < 0).
- contiguous: `entry[r, j]` = first workspace token `si*sbk + (ks % sbk if unaligned else 0)`, `-1` = skip;
  `entry[r, S]`, `entry[r, S+1]` = the row's valid token range `[max(ks,0), min(ke, num_kv))`.

# 2. Block top-k: the dense indexer layers without full-width logits

What it is: the same int32 top-k rows (and, for layer 20, the same candidate blocks) as today's dense path, computed
from fp32 block maxima `[rows, width / 8]` and the logits of the selected blocks `[rows, 16384]` instead of fp32 logits
`[rows, width]`.

## 2.1 What the dense path does today (image `vllm-moet-sm120:dsv41-64096a4`, read from source)

`sparse_attn_indexer` (`vllm/model_executor/layers/sparse_attn_indexer.py`), per indexer layer call, decode branch:
`fp8_fp4_paged_mqa_logits` -> fp32 `[rows, max_model_len / compress_ratio]`, then

| layers | pages | width at 1M | after the logits |
|---|---|---|---|
| 2, 8, 14 (ratio 2) | 64 rows | 524,288 (2 MiB/row) | `det_topk.topk_decode` |
| 20 (ratio 1, `candidate_write`) | 128 rows | 1,048,576 (4 MiB/row) | `select_candidate_blocks` (fp32 block scores `[rows, 131072]`), `det_topk.topk_decode` |
| 24, 28, 32, 36 (ratio 1, consumers) | 128 rows | 1,048,576 | `apply_candidate_mask` (uint8 flags `[rows, 131073]`, -inf written over the whole width), `det_topk.topk_decode` |

Rows: sm_120 flattens a request's 1 + `num_speculative_tokens` queries (`_supports_native_decode` is false for
next_n = 8), so every query is a row with its own context length and its own copy of the page table; `seq_lens` is
`[rows, 1]` (DeepGEMM asserts `context_lens.dim() == 2`).

`det_topk_sm120` (vLLM-Moet Dockerfile step 26): the k largest of a row's columns `[0, end)` under keys with every NaN above +inf and
-0 = +0; equal keys go to the lower column; output ascending columns, then -1; a row of at most k columns returns all
of them. `select_candidate_blocks` (vLLM-Moet Dockerfile step 31, `VLLM_MOET_CANDIDATE_DET_TOPK=1`): block score = NaN-propagating maximum
of the block's 8 logits, the newest block pinned to +inf (also over a NaN); `det_topk` with k = 2048 over the
`ceil(len / 8)` valid blocks; then the fixup: a selected block whose score is -inf becomes -1 (holes; NaN stays).

## 2.2 The design

    plain layer (2, 8, 14), S = 512 blocks        candidate source (20), S = 2048 blocks      consumer (24..36)
    scan      -> block maxima / short logits      scan (newest pinned, scores always)         compact  candidates -> sel
    det_topk  -> top S blocks, ascending          det_topk -> top S blocks                    rescore  their tokens
    rescore   -> logits of those blocks           rescore                                     det_topk
    det_topk  -> top 512 columns                  det_topk                                    finish   columns -> tokens
    finish    -> columns to tokens                finish   + the candidate rule (-inf -> -1)

- **scan** scores every token once (the dense kernels' fp32 value, bit for bit: same MMA, same relu / fma / add order,
  spelled with `_rn` intrinsics). A *long* row keeps only the maximum of each 8-token block; a *short* row keeps its
  logits (column = token), so it costs one pass, as in the dense path. Short = at most 16,384 columns for the source
  (its 2048 blocks), 65,536 for a plain layer (its scratch has the room; `VLLM_MOET_INDEXER_BLOCK_TOPK_SHORT[_SOURCE]`).
- **rescore** scores the tokens of the selected blocks of long rows again: column `j * 8 + o` = token `sel[j] * 8 + o`.
- The source's own top-512 comes from its 2048 candidate blocks *before* the -inf fixup.
- A consumer has no scan: only its candidate blocks' tokens are scored (the dense path masks everything else).

**Why it is exact.** Order tokens as `det_topk` does (key descending, then column ascending) and blocks by (maximum
key descending, block ascending). Take a token t of the top k, in block b, and suppose b is not among the top k
blocks: then k blocks precede b, and each holds a token whose key is its block's maximum - larger than b's maximum,
or equal to it in a lower block, hence at a lower column - so each of those k tokens precedes t, and t is not in the
top k. So the top k tokens lie in the top k blocks, for any k. The selected blocks are ascending, so column order is
token order and ties resolve the same way; a row of at most 512 tokens has all its blocks selected and returns all its
tokens, as the dense path does. For layer 20: pinning moves one block to the front, so a block of the unpinned top 512
is in the pinned top 513, inside the 2048 candidates.

Where equality is not guaranteed (never with finite logits):
- layer 20: its newest block holds a NaN *and* more than 1,535 other blocks have a +inf maximum (the pin then lowers
  the newest block's rank below 2048);
- a consumer: fewer than 512 of its candidate tokens are above -inf (the dense top-k then fills up with masked
  columns outside the candidates). Random UE8M0 scales over the whole byte range (10^5 inf / NaN logits) still gave
  identical rows in the tests.

## 2.3 Kernels (`sm120_sparse_indexer/csrc/block_topk_sm120.cu`)

- Every warp works alone on 16-token tiles (the m16n8k64 MXF4 MMA's A tile): the next tile's 1,024 K bytes go
  global -> shared with two 16-byte `cp.async` per lane into a private two-stage buffer while the current tile is
  scored from two `ldmatrix`; Q, its scales and the weights sit in registers; the tile's eight MMAs are issued k-major.
  No CTA barrier in the loop.
- Work split: a prefix sum of the rows' tile counts, built per CTA in shared memory; warp w of W takes the units
  `[total * w / W, total * (w + 1) / W)`. One launch shape for every mix of context lengths (graph-safe), balanced for
  one 1M row among 511 short ones.
- Row groups: the rows of a request read the same K. The dense kernel (and a scan row by row) fetch it once per row,
  so with more K than the 128 MiB L2 they are bound by DRAM. The scan splits the work by groups of 1 +
  `num_speculative_tokens` rows and walks a group in chunks of 8 tiles (128 tokens), each chunk for every row of the
  group in turn: the first row's fetch brings the chunk in, the others find it in L1 / L2. Grouping is a traversal
  order only (any group size gives the same rows); a launch with less than 64 MiB of K goes row by row.
- What bounds it: the instructions per tile and the warps resident per SM (24 at 80 registers per thread), not the
  MMAs. Measured on the way, scan of 128 rows x 1M: fragments by plain 4-byte global loads into registers 1.89 ms, by
  cp.async + ldmatrix with the same loop 1.99 ms (no gain by itself); then one page lookup per page instead of two
  per tile and masking only on a row's last tile 1.53 ms; 8 / 16 / 24 warps per SM 2.8 / 4.9 / 5.8 TB/s; a
  64-register cap (32 warps) 1.5x slower (spills); the Q operands passed by reference into the row loop -12 %.
- `det_topk_sm120` is called as it is (the image's, vLLM-Moet Dockerfile step 26), twice per call.

## 2.4 Results (GPU 3, RTX PRO 6000 Blackwell, image dsv41-64096a4; logs in `results/`, `./run_all.sh` reruns all)

**Exactness** (`tests/test_block_topk.py`, `results/test_block_topk_full.log`): 261 cases, all identical - top-k rows
and candidate blocks against the dense path, for the three layer kinds; decode at contexts 4k / 64k / 256k / 1M (ratio
2: half) x rows 1 / 8 / 37 / 128 / 512 at the engine's static widths; contexts around every threshold (512, 4,096,
16,384, 65,536); one batch mixing empty, tiny and 1M rows (503 rows); ties (pools of 2 and 3 pages, a constant cache);
inf / NaN logits; scan group sizes 1 / 3 / 8 / 64; prefill chunks of one and two requests (the second at an unaligned
offset), up to 9,000 rows (the host splits launches at 8,192 rows). `tests/test_vllm_block_patch.py`: vLLM's own
`SparseAttnIndexer` layer (the registered custom op, real metadata dataclasses, a mixed batch of 24 decode rows + two
prefill chunks) with the switch on vs off: identical buffers for the three kinds, block path taken, dense not run.
The section-1 suites still pass (53 + 7 cases).

**Memory** (`tests/graph_mem.py`, default allocator, max_model_len 1,048,576, the eight indexer calls of a decode step
captured in one CUDA graph): dense 6.000 MiB per row; consumers on the section-1 sparse API 6.000; block top-k
**0.570 MiB per row** (294 MiB at 512 rows against 3,072: -2.71 GiB per GPU). Every decode call allocates one scratch
block of the largest registered layer's size (`reserve_paged`: 0.572 MiB per row, the source's), so the eight calls
reuse one block of the graph pool; of it 0.5 MiB is the source's block maxima `[rows, 131072]` - it scales with
max_model_len, the rest (logits 64 KiB, selections 10 KiB) does not.

**Time per call**, CUDA-graph replays, microseconds, dense -> block (`results/bench_block_topk.log`; K pool 46 MiB =
L2-resident; dense at width = the context - see the caveat below):

| layer | context | 8 rows | 128 rows | 384 rows | 512 rows |
|---|---|---|---|---|---|
| 20 (source) | 64k | 34 -> 46 | 169 -> 184 | 528 -> 518 | 750 -> 675 |
| | 256k | 73 -> 72 | 821 -> 539 | 2,659 -> 1,561 | 3,362 -> 1,931 |
| | 1M | 197 -> 142 | 3,440 -> 1,781 | 10,378 -> 5,679 | 13,672 -> 7,706 |
| 2 / 8 / 14 (ratio 2) | 64k | 18 -> 26 | 85 -> 83 | 237 -> 230 | 301 -> 299 |
| | 256k | 32 -> 44 | 296 -> 269 | 1,091 -> 755 | 1,388 -> 936 |
| | 1M | 95 -> 83 | 1,465 -> 911 | 4,142 -> 2,742 | 5,385 -> 3,640 |
| 24..36 (consumer) | 64k | 28 -> 26 | 175 -> 67 | 531 -> 167 | 709 -> 208 |
| | 256k | 60 -> 24 | 724 -> 67 | 2,256 -> 169 | 2,914 -> 209 |
| | 1M | 192 -> 24 | 2,924 -> 65 | 8,647 -> 168 | 11,538 -> 210 |

Slower than dense: 8 rows below ~256k context (+3..+12 us per call) and any batch at 4k-16k context for the plain
and source layers (+1..+10 us): the block path is five graph nodes against the dense path's two (plain) or five
(source), and three of them do nothing on short rows. Not chased.

Caveat on the dense column: the bench runs the dense path at width = the context; the engine's width is
max_model_len, where `apply_candidate_mask` writes -inf over 1M columns per row and the block-score kernel reads
131,072 blocks per row whatever the context. The engine's dense consumers and source are therefore slower than this
table at short contexts (not measured here).

**Roofline** (K bytes scored per second = rows x context x 68 B / scan time; 1.54 TB/s achievable DRAM):
- K in the L2 (46 MiB pool), 128+ rows at 64k+ context: dense logits kernel 4.2-5.2 TB/s, scan 4.6-5.6 TB/s
  (5.9 row by row). Neither is DRAM-bound. Ceilings measured on an early build of the loop: MMAs alone (loads and
  epilogue removed) 13.7 TB/s; MMAs + the mandated relu / fma / add epilogue without loads 7.3 TB/s.
- More K than the L2 (`results/bench_block_topk_dram.log`: 64 requests x 1M tokens, each its own pages, 4.4 GiB):
  the dense logits kernel reads every row's K from DRAM: 1.39-1.40 TB/s = 91 % of the roofline, DRAM-bound
  (26.2 ms at 512 rows x 1M). The grouped scan scores 5.2 TB/s while 0.62 TB/s crosses DRAM (a request's K once, 40 %
  of the roofline): 7.0 ms. Whole call, 512 rows x 1M: layer 20 32.7 -> 8.1 ms, a ratio-2 layer 14.8 -> 4.1 ms.
  With the scan forced row by row (group 1): 1.44-1.46 TB/s, DRAM-bound like the dense kernel.

## 2.5 Integration (as served)

Nothing in the image changes; one file is overlaid and one directory mounted.

1. Artifacts (already built and current): `build/vllm_moet_block_topk_sm120/vllm_moet_block_topk_sm120.so`,
   `patch/patched/model_executor/layers/sparse_attn_indexer.py`. After any edit of `csrc/*.cu` run `./build.sh` (no GPU
   needed; the loader takes a prebuilt .so only if it is not older than its source, else it compiles at first use into
   `$TORCH_EXTENSIONS_DIR`). After a new image, regenerate the patched file:
   `python3 patch/patch_vllm_indexer_block_topk_sm120.py <copy of the image's vllm tree>` (anchors must match once).
2. Launch as today plus `EXTRA_DOCKER_ARGS="$(patch/mount_args_block_topk.sh)"`, which expands to
   - `-v <repo>/patch/patched/model_executor/layers/sparse_attn_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro`
   - `-v <repo>:/opt/vllm-moet/sm120_sparse_indexer:ro`
   - `-e VLLM_MOET_INDEXER_BLOCK_TOPK=1`
   That is decode only: prefill chunks stay on the dense code unless `-e VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL=1` is
   added (see "Prefill" below).
3. Needs, all in the served configuration: `VLLM_MOET_DET_TOPK=1` and `VLLM_MOET_CANDIDATE_DET_TOPK=1` (it reproduces
   their selections; without both it logs a warning and stays dense), `indexer_kv_dtype=mxfp4`, sm_120. Leave
   `attention_config.indexer_sparse_logits` off (not needed; it would also change the KV block stride alignment).
4. In the log, per worker: `vllm-moet: indexer block top-k from /opt/vllm-moet/sm120_sparse_indexer/sm120_sparse_indexer/block_topk.py
   (decode rows; prefill chunks dense)` - or `(decode rows and prefill chunks)` with the prefill switch.
   `vllm-moet: indexer block top-k does not take this decode layout (...)` (once) means a decode call fell back to
   the dense code - then the graph memory is not saved; `... unavailable, dense path: ...` means the module did not
   load.
5. Optional knobs: `VLLM_MOET_INDEXER_BLOCK_TOPK_SHORT_SOURCE=65536` makes layer 20 single-pass up to 64k context
   (+0.19 MiB per row); `VLLM_MOET_SPARSE_INDEXER_DIR` / `VLLM_MOET_SPARSE_INDEXER_BUILD` move the module / the build
   directory.

What the switch changes and what it does not:
- Decode (CUDA graphs): taken when the layout is one row per query (`next_n == 1`, not padded, `seq_lens` int32 with
  one entry per row, per-row page table), MXFP4 cache, fp32 weights, 8-token candidate blocks, no DCP. That is the
  sm_120 layout with or without speculative tokens. The op allocates one scratch tensor per call inside the capture
  (as the dense path allocates its logits) and launches a fixed set of kernels: scan, det_topk, rescore, det_topk,
  finish (consumer: compact, rescore, det_topk, finish).
- Prefill chunks (eager): **dense by default** since an engine run on 2026-11-10: with
  the block path on prefill chunks too, fresh prefill was slower than the dense engine - by 17 % at 15k, 26 % at 129k,
  19 % at 396k and 12 % at 713k tokens. The memory
  win needs decode only (prefill is eager, no graph holds its logits), so `VLLM_MOET_INDEXER_BLOCK_TOPK=1` now routes
  decode rows only and `VLLM_MOET_INDEXER_BLOCK_TOPK_PREFILL=1` adds the prefill chunks (`contiguous_*`; chunker and
  K gather unchanged; a TP rank's shard of rows passed as the dense path passes it). A step with decode rows and
  prefill chunks routes each part by its own switch; both write the dense path's bits (the candidate blocks are the
  same either way), so every mix is exact. With dense prefill the eager transient of the dense path is back: up to
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512 MiB) of logits per chunk plus 1/8 of it in block scores.
  Measured at module level afterwards (`bench/bench_prefill.py`, eager, wall clock, one GPU; one prefill step = the
  last 8,192 query tokens of one request, cut into chunks by the engine's 512 MiB rule; ms per step and layer, dense
  -> block, with the kernels the engine ran; every chunk's rows compared equal):

  | context | chunks x rows (ratio 1) | ratio-2 layer, one rank's 2,048 rows | source (20) | consumer (24..36) |
  |---|---|---|---|---|
  | 16k | 1 x 8,192 | 0.19 -> 0.21 | 1.81 -> 2.11 | 1.55 -> 3.02 |
  | 131k | 8 x 1,024 | 1.83 -> 2.04 | 23.4 -> 14.4 | 21.7 -> 2.9 |
  | 400k | 25 x 335 | 7.9 -> 4.7 | 103 -> 39.8 | 87 -> 3.3 |
  | 700k | 43 x 191 | 18.5 -> 8.2 | 242 -> 67.3 | 232 -> 3.8 |

  Groups of 256 rows (the size of the prefill's replay graphs): block faster at every context, 1.3x (16k) to 9.6x
  (700k) over the eight layers. Host time per step (until the last launch returned) is lower for block than for dense
  everywhere (700k source: 2.3 vs 4.7 ms). So the chunk cost does **not** reproduce the engine's slowdown: per chunk
  the block path is slower only at 16k (and by a millisecond or two per step), faster from 131k up. What made the
  engine's prefill 12-26 % slower is therefore not the indexer chunks' compute or launch cost as measured alone; not
  found. Not measured: the same chunks inside the engine (pipelined groups on two streams, the phase / replay graphs
  at their captured width, the engine's memory pressure). The boot with dense prefill + block decode tells whether the
  slowdown follows the prefill path or the engine configuration.
  The one real defect the bench shows - consumers at 8,192 rows per chunk, 3.02 ms against 1.55 - is fixed on branch
  `prefill-switch`, not merged here: the work split kept a prefix entry per
  row in shared memory (33 KiB at 8,192 rows, which by the ~100 KiB per SM leaves the rescore 4 warps per SM instead
  of 22); there it is per bucket of rows, 2 KiB whatever the row count (16k consumer 3.02 -> 1.68 ms, the step 14.9 ->
  9.4 ms against dense 8.5; other contexts unchanged; decode unchanged within 3 %, consumers at 512 rows 208 -> 185
  us; 279 exactness cases and the three-mode op test identical; 0.570 MiB per row). The occupancy query it also
  caches is not a measurable cost (host times equal on both builds).
- max_model_len: nothing assumes 1,048,576; the scratch is `rows x (width / 8 + short + S + 512 + 3) x 4` B with
  width = the layer's max_model_len. Limits: width < 2^30, at most 8,192 rows per launch (larger batches are split on
  the host), page size a power of two >= 16, K pages 16-byte aligned (true for the 512 / 576 B aligned vLLM pages).
- The profile run's `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` reservation and the prefill chunk budget are untouched.
- Not covered: FP8 indexer cache, native next_n > 1 rows, DCP / PCP, candidate_block_size != 8 - those calls stay dense.

## 2.6 Not done

- Engine-level results are in the README (measured separately from this module).
- Short-context small-batch calls are a few microseconds slower (above); fusing the column top-k with the remap
  would remove one node but needs a copy of det_topk.
- The grouped scan pays one un-overlapped fetch and a Q reload per 8-tile run (-9 % when K is L2-resident, 5.4 vs
  5.9 TB/s); touching the next chunk ahead and carrying the fetch across runs should recover most of it.
- A hierarchical selection (64-token maxima first) would cut the source's block maxima from 0.5 to ~0.13 MiB per
  row at one more pass over 12.5 % of the tokens.
