// Exact indexer top-k without full-width logits, for the DeepSeek-V4.1 MXFP4 indexer cache on sm_120.
//
// What it is: the three kernels between the indexer cache and the top-k that replace a dense fp32 [rows, context]
// logits buffer by fp32 block maxima [rows, context / 8] and fp32 logits of the selected blocks [rows, S * 8]:
//
//   scan     every token of a row scored once (the value DeepGEMM's dense sm_120 kernels compute, bit for bit);
//            long rows (more than `short_blocks` blocks) keep only each 8-token block's maximum, short rows keep
//            their logits (column = token), so a short row is scored once, as in the dense path.
//   rescore  the tokens of the S blocks selected per long row scored again: column j * 8 + o = token sel[j] * 8 + o.
//   finish   selected columns -> token positions; the candidate blocks' -inf -> -1 rule (vLLM's step 31 fixup).
//
//   compact  a candidate consumer's blocks (the candidate source's output: ascending, -1 where dropped) packed to the
//            front; its rescore then scores those blocks only (no scan: the dense path masks every other token).
//
// Between them the caller runs det_topk_sm120 twice (over the block maxima, over the logits). Why it is exact: with
// values ordered as det_topk orders them (NaN above +inf, -0 = +0, equal values to the lower column), a token of the
// top k has at most k - 1 tokens before it; a block not in the top k blocks (ordered by its maximum, equal maxima to the
// lower block) has k blocks before it, each holding a token that is before every token of that block. So the top k
// tokens lie in the top k blocks, and since the selected blocks ascend, column order is token order.
//
// A token's value (as in sparse_mqa_logits_sm120.cu):
//   logit = (P_0 + P_1) + (P_2 + P_3),  P_t = (((0 + T_0) + T_1) + T_2) + T_3,
//   T_nt = fma(relu(d[h0]), w[h0], relu(d[h0 + 1]) * w[h0 + 1]),  h0 = nt * 8 + 2t,
//   d[h] = the m16n8k64 MXF4 block-scaled MMA of the token's K row against head h's Q row, two k-steps of 64.
// Every warp works alone on 16-token tiles: the next tile's 1024 K bytes go global -> shared with two cp.async per
// lane (a private two-stage buffer per warp, no CTA barrier) while the current tile is scored from ldmatrix fragments;
// Q, its scales and the weights sit in registers (lane (g, t) holds bytes 4t.. of each 16-byte quarter of head
// nt * 8 + g). The eight MMAs of a tile are issued k-major so that the four n-tiles' accumulators overlap. What bounds
// the loop is its instruction count per tile and the warps resident per SM, not the MMAs (NOTES.md, section 2.3).
//
// Work is split evenly over the warps of the launch by token count (a prefix sum of the rows' tile counts, built per
// CTA), so one launch shape serves every mix of context lengths: graph-safe.
//
// Decode rows come in groups that read the same K (sm_120 flattens a request's 1 + num_speculative_tokens queries into
// rows with their own copy of its page table). The dense kernel, and a scan row by row, fetch that K once per row:
// with more K than the 128 MiB L2 holds the scan is then bound by DRAM (1.4-1.5 TB/s of the 1.54 achievable). So the
// scan splits the work by row *groups* and a warp walks its part of a group in chunks of a few tiles, each chunk for
// every row of the group in turn: the first row's fetch brings the chunk in, the others find it in L1 / L2. Grouping is
// only a traversal order - rows that share nothing are scored just the same. A launch with less K than half the L2 is
// scanned row by row (decided on the device from the rows' lengths).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kHeads = 32;     // index_n_heads
constexpr int kRowBytes = 64;  // MXFP4 head_dim 128: 64 bytes of e2m1 pairs (4 UE8M0 scale bytes elsewhere)
constexpr int kBlock = 8;      // tokens per block
constexpr int kMaxRows = 8192; // rows per launch (the per-CTA prefix sum); the host splits larger batches

__device__ __forceinline__ void fp4_mma_block_scaled(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                     uint16_t sfa, uint16_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(static_cast<uint32_t>(sfa)),
          "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)), "r"(static_cast<uint32_t>(sfb)),
          "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ uint16_t sf_pair(uint32_t packed, uint32_t first_byte) {
    return static_cast<uint16_t>((packed >> (first_byte * 8)) & 0xFFFF);
}

// One row's B operands for the four n-tiles (heads nt * 8 + g), Q scales and weights, as lane (g, t) holds them.
struct QOp {
    uint32_t b[4][2][2];
    uint32_t sf[4];
    float w0[4], w1[4];
};

__device__ __forceinline__ void load_q(QOp& q, const uint8_t* qrow, const uint32_t* sfrow, const float* wrow, int g,
                                       int t) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const uint32_t* h = reinterpret_cast<const uint32_t*>(qrow + (nt * 8 + g) * kRowBytes) + t;
        q.b[nt][0][0] = __ldg(h);
        q.b[nt][0][1] = __ldg(h + 4);
        q.b[nt][1][0] = __ldg(h + 8);
        q.b[nt][1][1] = __ldg(h + 12);
        q.sf[nt] = __ldg(sfrow + nt * 8 + g);
        q.w0[nt] = __ldg(wrow + nt * 8 + t * 2);
        q.w1[nt] = __ldg(wrow + nt * 8 + t * 2 + 1);
    }
}

// A 16-token tile's A operands for the two k-steps, and its scale word (token g + (t & 1) * 8's).
struct AFrag {
    uint32_t a[2][4];
    uint32_t sf;
};

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// CuTe Swizzle<2, 4, 3> (64-byte rows): bits [4:5] ^= bits [7:8], so the ldmatrix rows of a tile hit distinct banks.
__device__ __forceinline__ int swz64(int x) { return x ^ ((x & 0x180) >> 3); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&d)[4], const void* smem) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "l"(__cvta_generic_to_shared(smem))
                 : "memory");
}

// The fragments of a swizzled 16 x 64-byte tile: lane row (lane & 7) + ((lane >> 3) & 1) * 8, column
// (lane >> 4) * 16 + k * 32 (DeepGEMM's load_a_fragment<64>).
__device__ __forceinline__ void load_a(AFrag& f, const uint8_t* tile, int lane) {
    const int base = ((lane & 7) + ((lane >> 3) & 1) * 8) * kRowBytes;
    const int xr = swz64(base) ^ base;
    ldmatrix_x4(f.a[0], tile + base + (((lane >> 4) * 16) ^ xr));
    ldmatrix_x4(f.a[1], tile + base + ((((lane >> 4) * 16) + 32) ^ xr));
}

// v0 = the logit of token g, v1 = of token g + 8 (the same in the four lanes of a quad).
__device__ __forceinline__ void score16(const AFrag& f, const QOp& q, float& v0, float& v1) {
    float d[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            fp4_mma_block_scaled(d[nt], f.a[k], q.b[nt][k][0], q.b[nt][k][1], sf_pair(f.sf, k * 2),
                                 sf_pair(q.sf[nt], k * 2));
    }
    float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const float t0 = __fmaf_rn(fmaxf(d[nt][0], 0.0f), q.w0[nt], __fmul_rn(fmaxf(d[nt][1], 0.0f), q.w1[nt]));
        const float t1 = __fmaf_rn(fmaxf(d[nt][2], 0.0f), q.w0[nt], __fmul_rn(fmaxf(d[nt][3], 0.0f), q.w1[nt]));
        p0 = __fadd_rn(p0, t0);
        p1 = __fadd_rn(p1, t1);
    }
#pragma unroll
    for (int s = 1; s <= 2; s <<= 1) {
        p0 = __fadd_rn(p0, __shfl_xor_sync(0xffffffffu, p0, s));
        p1 = __fadd_rn(p1, __shfl_xor_sync(0xffffffffu, p1, s));
    }
    v0 = p0;
    v1 = p1;
}

__device__ __forceinline__ float max_nan(float a, float b) { return (a != a) ? a : ((b != b) ? b : fmaxf(a, b)); }

struct Params {
    const uint8_t* q;        // [rows, 32, 64] packed e2m1
    const uint32_t* sf_q;    // [rows, 32] packed UE8M0
    const float* w;          // [rows, 32]
    const uint8_t* kv;       // paged: page 0; contiguous: [num_kv, 64]
    const uint32_t* sf_kv;   // contiguous: [num_kv]
    const int32_t* lens;     // paged: context length per row
    const int32_t* ks;       // contiguous: [ks, ke) per row
    const int32_t* ke;
    const int32_t* block_table;  // paged: [rows, bt_cols]
    float* scores;           // [rows, width_blocks]
    float* logits;           // [rows, logits_width]
    int32_t* valid_blocks;   // [rows] ceil(len / 8) (rescore by_count: the row's selected blocks)
    int32_t* sel_ends;       // [rows] scan: the block scores det_topk may read (0 for a short row without scores)
    int32_t* end;            // [rows]
    const int32_t* sel;      // rescore: [rows, S] selected blocks, ascending
    int64_t q_stride, sfq_stride, w_stride;  // bytes, words, floats
    int64_t page_stride, num_pages, num_kv, bt_stride, bt_cols;
    int64_t scores_stride, logits_stride, sel_stride;
    int32_t rows, width_tokens, width_blocks, logits_width, S, page_shift;
    int32_t short_blocks;    // rows of at most this many blocks are short (>= S)
    int32_t group, chunk;    // scan: rows per group, tiles per chunk (see the header)
    int32_t group_min_units; // scan: grouped only when the groups' tiles (a KiB of K each) exceed this
    bool pin, always_scores;
    bool by_count;           // rescore: valid_blocks[r] = the number of selected blocks of row r (a candidate consumer)
};

template <bool kPaged>
__device__ __forceinline__ int32_t row_len(const Params& p, int r) {
    int64_t n;
    if constexpr (kPaged) {
        n = p.lens[r];
    } else {
        // [ks, ke) lies in the workspace; a row that does not is cut to it (the tile loads rely on it)
        const int64_t ks = p.ks[r];
        n = static_cast<int64_t>(p.ke[r]) - ks;
        const int64_t room = ks >= 0 ? p.num_kv - ks : 0;
        n = n > room ? room : n;
    }
    n = n < 0 ? 0 : n;
    return static_cast<int32_t>(n > p.width_tokens ? p.width_tokens : n);
}

// An 8-token group of a row: where its K rows and scale words are.
struct Group {
    const uint8_t* rows;  // paged: the group's first K row (a valid page, else page 0's)
    const uint32_t* sf;   // paged: its first scale word
    int64_t tok;          // contiguous: its first workspace token (may run past the workspace)
    bool ok;              // paged: the page exists; contiguous: the group is given
};

// The group of row r starting at row-local token `local` (a multiple of 8; negative: no group).
template <bool kPaged>
__device__ __forceinline__ Group locate(const Params& p, int r, int64_t local, int64_t base) {
    Group gr;
    if constexpr (kPaged) {
        const int64_t lp = local >> p.page_shift;
        const int64_t page = (local >= 0 && lp < p.bt_cols) ? p.block_table[r * p.bt_stride + lp] : -1;
        gr.ok = page >= 0 && page < p.num_pages;
        const uint8_t* pg = p.kv + (gr.ok ? page : 0) * p.page_stride;
        const int64_t off = gr.ok ? (local & ((int64_t(1) << p.page_shift) - 1)) : 0;
        gr.rows = pg + off * kRowBytes;
        gr.sf = reinterpret_cast<const uint32_t*>(pg + (int64_t(1) << p.page_shift) * kRowBytes) + off;
        gr.tok = 0;
    } else {
        gr.ok = local >= 0;
        gr.tok = base + (local >= 0 ? local : 0);
        gr.rows = nullptr;
        gr.sf = nullptr;
    }
    return gr;
}

// Token j of the group can be read (contiguous: inside the workspace).
template <bool kPaged>
__device__ __forceinline__ bool readable(const Params& p, const Group& gr, int j) {
    if constexpr (kPaged) return gr.ok;
    else return gr.ok && gr.tok + j >= 0 && gr.tok + j < p.num_kv;
}

// All eight tokens of the group can be read.
template <bool kPaged>
__device__ __forceinline__ bool whole(const Params& p, const Group& gr) {
    if constexpr (kPaged) return gr.ok;
    else return gr.ok && gr.tok >= 0 && gr.tok + kBlock <= p.num_kv;
}

// Lane's 16-byte chunk of the group (token lane >> 2, quarter lane & 3); an unreadable token reads token 0.
template <bool kPaged>
__device__ __forceinline__ const uint8_t* chunk_src(const Params& p, const Group& gr, int lane) {
    if constexpr (kPaged) {
        return gr.rows + lane * 16;
    } else {
        const int64_t tok = gr.tok + (lane >> 2);
        return p.kv + ((gr.ok && tok >= 0 && tok < p.num_kv) ? tok : 0) * kRowBytes + (lane & 3) * 16;
    }
}

template <bool kPaged>
__device__ __forceinline__ const uint32_t* scale_src(const Params& p, const Group& gr, int g) {
    if constexpr (kPaged) {
        return gr.sf + g;
    } else {
        const int64_t tok = gr.tok + g;
        return p.sf_kv + ((gr.ok && tok >= 0 && tok < p.num_kv) ? tok : 0);
    }
}

// Starts the copy of a tile (groups a: tokens g, b: tokens g + 8) into a stage; returns its scale word for this lane.
template <bool kPaged>
__device__ __forceinline__ uint32_t fetch_tile(const Params& p, const Group& ga, const Group& gb, uint8_t* stage,
                                               int lane) {
    const int dst = swz64(lane * 16);
    cp_async16(stage + dst, chunk_src<kPaged>(p, ga, lane));
    cp_async16(stage + 512 + dst, chunk_src<kPaged>(p, gb, lane));
    cp_async_commit();
    return __ldg((lane & 1) ? scale_src<kPaged>(p, gb, lane >> 2) : scale_src<kPaged>(p, ga, lane >> 2));
}

// The maximum of m over the eight quads' lanes of equal t (NaN-propagating; even t: tokens g, odd t: tokens g + 8).
__device__ __forceinline__ float block_max(float m) {
    if (__ballot_sync(0xffffffffu, m != m) == 0) {
#pragma unroll
        for (int s = 4; s <= 16; s <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    } else {
#pragma unroll
        for (int s = 4; s <= 16; s <<= 1) m = max_nan(m, __shfl_xor_sync(0xffffffffu, m, s));
    }
    return m;
}

constexpr int kStageBytes = 1024;                 // a 16-token tile
constexpr int kWarpSmem = 2 * kStageBytes;        // two stages per warp

// ---------------------------------------------------------------------------------------------------------------
// The launch's work split. s_pre[0..n] = the exclusive prefix sum of count(i) (s_pre[n] = the total), built by every
// CTA (s_ws: 32 words); a warp takes the units [total * w / W, total * (w + 1) / W) of the W warps of the launch.

template <typename F>
__device__ __forceinline__ void block_prefix(int n, uint32_t* s_pre, uint32_t* s_ws, F count) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, T = blockDim.x;
    const int per = (n + T - 1) / T;
    uint32_t sum = 0;
    for (int j = 0; j < per; ++j) {
        const int i = tid * per + j;
        if (i < n) sum += count(i);
    }
    uint32_t x = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_ws[warp] = x;
    __syncthreads();
    uint32_t run = x - sum;
    for (int w = 0; w < warp; ++w) run += s_ws[w];
    for (int j = 0; j < per; ++j) {
        const int i = tid * per + j;
        if (i < n) {
            s_pre[i] = run;
            run += count(i);
            if (i == n - 1) s_pre[n] = run;
        }
    }
    __syncthreads();
}

// This warp's units [u, hi) of s_pre[n]; false when it has none.
__device__ __forceinline__ bool warp_units(const uint32_t* s_pre, int n, uint32_t& u, uint32_t& hi) {
    const uint64_t total = s_pre[n];
    const uint64_t nw = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t wid = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    u = static_cast<uint32_t>(total * wid / nw);
    hi = static_cast<uint32_t>(total * (wid + 1) / nw);
    return u < hi;
}

// The item holding unit u < s_pre[n]: the largest i with s_pre[i] <= u.
__device__ __forceinline__ int prefix_find(const uint32_t* s_pre, int n, uint32_t u) {
    int lo = 0, hi = n;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (s_pre[mid] <= u) lo = mid;
        else hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------------------------------------------
// Tiles [t_begin, t_end) of row r in token order (tile ti = tokens [16 ti, 16 ti + 16)): logits to lrow (column =
// token) and / or block maxima to srow. n = the row's length, s_stage = this warp's two tile stages.
//
// The fetch stream runs one tile ahead of the scoring: f_rows = this lane's 16-byte chunk of the tile's first group
// (the second is 512 bytes on), f_sf = this lane's scale word, f_ok = the tile's memory exists. A paged tile lies in one
// page (pages are multiples of 16 tokens); a contiguous row's tiles are consecutive workspace rows, of which only the
// row's last tile can run past the workspace.

template <bool kPaged>
__device__ __forceinline__ void score_run(const Params& p, int r, uint32_t t_begin, uint32_t t_end, int32_t n,
                                          bool write_logits, bool write_scores, uint8_t* s_stage) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    QOp q;
    load_q(q, p.q + r * p.q_stride, p.sf_q + r * p.sfq_stride, p.w + r * p.w_stride, g, t);
    // this lane's places in a tile: its cp.async chunk, its two ldmatrix rows, its scale word
    const int dst_lane = swz64(lane * 16);
    const int a_base = ((lane & 7) + ((lane >> 3) & 1) * 8) * kRowBytes;
    const int a_lane0 = a_base + (((lane >> 4) * 16) ^ (swz64(a_base) ^ a_base));
    const int a_lane1 = a_base + ((((lane >> 4) * 16) + 32) ^ (swz64(a_base) ^ a_base));
    const int sf_lane = g + (t & 1) * 8;
    const uint32_t tile_mask = kPaged ? (1u << (p.page_shift - 4)) - 1 : 0;
    const int32_t nb = (n + kBlock - 1) / kBlock;
    const int32_t newest = (n - 1) / kBlock;
    const int64_t base = kPaged ? 0 : p.ks[r];
    float* srow = p.scores + r * p.scores_stride;
    float* lrow = p.logits + r * p.logits_stride;
    const uint32_t row_tiles = static_cast<uint32_t>(n + 15) >> 4;
    const uint8_t* f_rows = nullptr;
    const uint32_t* f_sf = nullptr;
    bool f_ok = true;
    const auto seek = [&](uint32_t ti) {
        if constexpr (kPaged) {
            const int tile_shift = p.page_shift - 4;
            const int64_t lp = ti >> tile_shift;
            const int64_t page = lp < p.bt_cols ? p.block_table[r * p.bt_stride + lp] : -1;
            f_ok = page >= 0 && page < p.num_pages;
            const uint8_t* pg = p.kv + (f_ok ? page : 0) * p.page_stride;
            const int64_t off = f_ok ? static_cast<int64_t>(ti & ((1u << tile_shift) - 1)) * 16 : 0;
            f_rows = pg + off * kRowBytes + lane * 16;
            f_sf = reinterpret_cast<const uint32_t*>(pg + (int64_t(1) << p.page_shift) * kRowBytes) + off + sf_lane;
        } else {
            const int64_t tok = base + static_cast<int64_t>(ti) * 16;
            f_rows = p.kv + tok * kRowBytes + lane * 16;
            f_sf = p.sf_kv + tok + sf_lane;
        }
    };
    const auto fetch = [&](uint32_t ti, uint8_t* stage) -> uint32_t {
        if constexpr (!kPaged) {
            if (ti + 1 == row_tiles) {  // the row's last tile: clamp every token into the workspace
                const Group ga = locate<kPaged>(p, r, static_cast<int64_t>(ti) * 16, base);
                const Group gb = locate<kPaged>(p, r, static_cast<int64_t>(ti) * 16 + 8, base);
                return fetch_tile<kPaged>(p, ga, gb, stage, lane);
            }
        }
        cp_async16(stage + dst_lane, f_rows);
        cp_async16(stage + 512 + dst_lane, f_rows + 512);
        cp_async_commit();
        return __ldg(f_sf);
    };
    seek(t_begin);
    bool cur_ok = f_ok, nxt_ok = true;
    uint32_t cur_sf = fetch(t_begin, s_stage), nxt_sf = 0;
    uint8_t* stage = s_stage;
    for (uint32_t ti = t_begin; ti < t_end; ++ti) {
        uint8_t* other = s_stage + (stage == s_stage ? kStageBytes : 0);
        if (ti + 1 < t_end) {
            if constexpr (kPaged) {
                if (((ti + 1) & tile_mask) == 0) {
                    seek(ti + 1);
                } else {
                    f_rows += kStageBytes;
                    f_sf += 16;
                }
            } else {
                f_rows += kStageBytes;
                f_sf += 16;
            }
            nxt_ok = f_ok;
            nxt_sf = fetch(ti + 1, other);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        AFrag f;
        ldmatrix_x4(f.a[0], stage + a_lane0);
        ldmatrix_x4(f.a[1], stage + a_lane1);
        f.sf = cur_sf;
        float v0, v1;
        score16(f, q, v0, v1);
        const int64_t tok_a = static_cast<int64_t>(ti) * 16 + g, tok_b = tok_a + 8;
        const bool last = ti + 1 == row_tiles;  // the only tile with tokens past the row or the newest block
        if (last || !cur_ok) {
            v0 = (cur_ok && tok_a < n) ? v0 : -INFINITY;
            v1 = (cur_ok && tok_b < n) ? v1 : -INFINITY;
        }
        if (write_logits) {
            if (t == 0 && tok_a < p.logits_width) lrow[tok_a] = v0;
            if (t == 1 && tok_b < p.logits_width) lrow[tok_b] = v1;
        }
        if (write_scores) {
            float m = block_max((t & 1) ? v1 : v0);  // even t: block 2 ti, odd t: block 2 ti + 1
            if (lane < 2) {
                const int64_t b = static_cast<int64_t>(ti) * 2 + lane;
                if (last && p.pin && b == newest) m = INFINITY;
                if (b < nb) srow[b] = m;
            }
        }
        stage = other;
        cur_ok = nxt_ok;
        cur_sf = nxt_sf;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// scan

// No register cap: at its ~80 registers per thread 24 warps fit an SM; capped to 64 (32 warps) the spills cost 1.5x.
template <bool kPaged>
__global__ void __launch_bounds__(256) scan_kernel(const Params p) {
    // dynamic shared memory: the warps' tile stages, then the prefix sum of the groups' tile counts and the warps' sums
    extern __shared__ __align__(128) uint8_t s_dyn[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = blockDim.x;
    const int rows = p.rows;
    int G = p.group;
    int ngroups = (rows + G - 1) / G;
    uint32_t* s_pre = reinterpret_cast<uint32_t*>(s_dyn + (T >> 5) * kWarpSmem);
    uint32_t* s_ws = s_pre + rows + 1;
    uint8_t* s_stage = s_dyn + warp * kWarpSmem;
    if (blockIdx.x == 0) {
        for (int r = tid; r < rows; r += T) {
            const int32_t n = row_len<kPaged>(p, r);
            const int32_t nb = (n + kBlock - 1) / kBlock;
            p.valid_blocks[r] = nb;
            p.sel_ends[r] = (nb > p.short_blocks || p.always_scores) ? nb : 0;
            p.end[r] = n;
        }
    }
    // a group's units: the tiles of its longest row (each stands for that tile of every row of the group)
    const auto group_tiles = [&](int gi) {
        uint32_t m = 0;
        for (int r = gi * G; r < rows && r < (gi + 1) * G; ++r) {
            const uint32_t tiles = static_cast<uint32_t>(row_len<kPaged>(p, r) + 15) >> 4;
            m = tiles > m ? tiles : m;
        }
        return m;
    };
    block_prefix(ngroups, s_pre, s_ws, group_tiles);
    if (G > 1 && s_pre[ngroups] <= static_cast<uint32_t>(p.group_min_units)) {
        // so little K (a KiB per unit) that the L2 holds it all: nothing to share, and row by row every warp gets
        // its own tiles instead of a few warps walking whole groups. The same decision in every CTA.
        G = 1;
        ngroups = rows;
        block_prefix(ngroups, s_pre, s_ws, group_tiles);
    }
    uint32_t u, hi;
    if (!warp_units(s_pre, ngroups, u, hi)) return;
    int gi = prefix_find(s_pre, ngroups, u);
    while (u < hi) {
        const uint32_t g0 = s_pre[gi], g1 = s_pre[gi + 1];
        const uint32_t a = u - g0, b = (hi < g1 ? hi : g1) - g0;
        const int r_begin = gi * G, r_end = rows < r_begin + G ? rows : r_begin + G;
        // one row: the whole range in one run; several: chunk by chunk, every row of the group per chunk
        const uint32_t step = r_end - r_begin > 1 ? static_cast<uint32_t>(p.chunk) : b - a;
        for (uint32_t c0 = a; c0 < b; c0 += step) {
            const uint32_t c1 = c0 + step < b ? c0 + step : b;
            for (int r = r_begin; r < r_end; ++r) {
                const int32_t n = row_len<kPaged>(p, r);
                const uint32_t row_tiles = static_cast<uint32_t>(n + 15) >> 4;
                if (c0 >= row_tiles) continue;
                const bool short_row = (n + kBlock - 1) / kBlock <= p.short_blocks;
                score_run<kPaged>(p, r, c0, c1 < row_tiles ? c1 : row_tiles, n, short_row,
                                  !short_row || p.always_scores, s_stage);
            }
        }
        u = g0 + b;
        if (u < hi) {
            ++gi;
            while (s_pre[gi + 1] == s_pre[gi]) ++gi;  // u < hi <= total: a group with tiles follows
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// rescore

template <bool kPaged>
__global__ void __launch_bounds__(256) rescore_kernel(const Params p) {
    // dynamic shared memory: the warps' tile stages, then the prefix sum of the rows' tile counts and the warps' sums
    extern __shared__ __align__(128) uint8_t s_dyn[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int T = blockDim.x;
    const int rows = p.rows;
    uint32_t* s_pre = reinterpret_cast<uint32_t*>(s_dyn + (T >> 5) * kWarpSmem);
    uint32_t* s_ws = s_pre + rows + 1;
    uint8_t* s_stage = s_dyn + warp * kWarpSmem;
    const uint32_t tiles_per_row = static_cast<uint32_t>(p.S + 1) / 2;
    // a row's tiles (tile ti = selected blocks 2 ti, 2 ti + 1): a candidate consumer's selected blocks (by_count:
    // valid_blocks = their number, whatever the row's length; compact wrote its end); else all S blocks of a long row
    // and none of a short one (the scan wrote its logits and its end)
    block_prefix(rows, s_pre, s_ws, [&](int r) {
        const int32_t nb = p.valid_blocks[r];
        if (p.by_count) return static_cast<uint32_t>(nb + 1) >> 1;
        return nb > p.short_blocks ? tiles_per_row : 0u;
    });
    uint32_t u, hi;
    if (!warp_units(s_pre, rows, u, hi)) return;
    int r = prefix_find(s_pre, rows, u) - 1;
    while (u < hi) {
        ++r;
        while (s_pre[r + 1] <= u) ++r;  // u < hi <= total: the row holding unit u follows
        const uint32_t row0 = s_pre[r], row1 = s_pre[r + 1];
        const uint32_t t_begin = u - row0, t_stop = (hi < row1 ? hi : row1) - row0;
        u = row0 + t_stop;
        const int32_t nb = p.valid_blocks[r];
        const int32_t n = row_len<kPaged>(p, r);
        if (p.by_count && nb == (n + kBlock - 1) / kBlock) {
            // every block of the row is selected (ascending, distinct): the columns are the row's tokens in order
            score_run<kPaged>(p, r, t_begin, t_stop, n, true, false, s_stage);
            continue;
        }
        const int64_t base = kPaged ? 0 : p.ks[r];
        const int32_t* sel = p.sel + r * p.sel_stride;
        float* lrow = p.logits + r * p.logits_stride;
        if (!p.by_count && t_begin == 0 && lane == 0) {
            // every one of the S selected blocks is valid and they ascend: only the last can be the partial newest
            const int64_t last = sel[p.S - 1];
            int64_t fill = static_cast<int64_t>(n) - last * kBlock;
            fill = fill < 0 ? 0 : (fill > kBlock ? kBlock : fill);
            p.end[r] = static_cast<int32_t>(static_cast<int64_t>(p.S - 1) * kBlock + fill);
        }
        QOp q;
        load_q(q, p.q + r * p.q_stride, p.sf_q + r * p.sfq_stride, p.w + r * p.w_stride, g, t);

        // tile ti = selected blocks 2 ti (tokens g) and 2 ti + 1 (tokens g + 8); la / lb: their first row-local tokens
        int64_t cur_la, cur_lb, nxt_la = -1, nxt_lb = -1;
        const auto groups = [&](uint32_t ti, int64_t& la, int64_t& lb, Group& ga, Group& gb) {
            const int32_t ba = sel[2 * ti];
            const int32_t bb = (2 * ti + 1 < static_cast<uint32_t>(p.S)) ? sel[2 * ti + 1] : -1;
            la = ba >= 0 ? static_cast<int64_t>(ba) * kBlock : -1;
            lb = bb >= 0 ? static_cast<int64_t>(bb) * kBlock : -1;
            ga = locate<kPaged>(p, r, la, base);
            gb = locate<kPaged>(p, r, lb, base);
        };
        Group cur_a, cur_b;
        groups(t_begin, cur_la, cur_lb, cur_a, cur_b);
        Group nxt_a = cur_a, nxt_b = cur_b;
        uint32_t cur_sf = fetch_tile<kPaged>(p, cur_a, cur_b, s_stage, lane), nxt_sf = 0;
        for (uint32_t ti = t_begin; ti < t_stop; ++ti) {
            const uint8_t* stage = s_stage + ((ti - t_begin) & 1) * kStageBytes;
            if (ti + 1 < t_stop) {
                groups(ti + 1, nxt_la, nxt_lb, nxt_a, nxt_b);
                nxt_sf = fetch_tile<kPaged>(p, nxt_a, nxt_b, s_stage + ((ti + 1 - t_begin) & 1) * kStageBytes, lane);
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            AFrag f;
            load_a(f, stage, lane);
            f.sf = cur_sf;
            float v0, v1;
            score16(f, q, v0, v1);
            v0 = (cur_la >= 0 && cur_la + g < n && readable<kPaged>(p, cur_a, g)) ? v0 : -INFINITY;
            v1 = (cur_lb >= 0 && cur_lb + g < n && readable<kPaged>(p, cur_b, g)) ? v1 : -INFINITY;
            const int64_t col = static_cast<int64_t>(ti) * 16 + g;
            if (t == 0 && col < p.logits_width) lrow[col] = v0;
            if (t == 1 && col + 8 < p.logits_width) lrow[col + 8] = v1;
            cur_a = nxt_a;
            cur_b = nxt_b;
            cur_la = nxt_la;
            cur_lb = nxt_lb;
            cur_sf = nxt_sf;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// compact: a candidate consumer's row. cand[r, :K] = candidate blocks in ascending order with -1 where the source
// dropped one (det_topk's ascending columns after the -inf rule). sel[r, :] = the blocks inside the row
// (0 <= block < ceil(len / 8)) in that order, then -1; count[r] = how many; end[r] = the number of their token columns
// inside the row (only the last block, the newest, can be partial).

struct CompactParams {
    const int32_t* cand;
    const int32_t* lens;
    const int32_t* ks;
    const int32_t* ke;
    int32_t* sel;
    int32_t* count;
    int32_t* end;
    int64_t cand_stride, sel_stride;
    int32_t K, S, width_tokens;
};

template <bool kPaged>
__global__ void compact_kernel(const CompactParams p) {
    __shared__ uint32_t s_ws[33];
    __shared__ int32_t s_last[32];
    const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = blockDim.x, nwarp = T >> 5;
    int64_t n64 = kPaged ? static_cast<int64_t>(p.lens[r]) : static_cast<int64_t>(p.ke[r]) - p.ks[r];
    n64 = n64 < 0 ? 0 : (n64 > p.width_tokens ? p.width_tokens : n64);
    const int32_t n = static_cast<int32_t>(n64);
    const int32_t nb = (n + kBlock - 1) / kBlock;
    const int32_t* cand = p.cand + r * p.cand_stride;
    int32_t* sel = p.sel + r * p.sel_stride;
    const int per = (p.K + T - 1) / T;
    uint32_t c = 0;
    int32_t last = -1;
    for (int j = 0; j < per; ++j) {
        const int i = tid * per + j;
        const int32_t e = i < p.K ? cand[i] : -1;
        if (e >= 0 && e < nb) {
            ++c;
            last = e > last ? e : last;
        }
    }
    uint32_t x = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    last = __reduce_max_sync(0xffffffffu, last);
    if (lane == 31) s_ws[warp] = x;
    if (lane == 0) s_last[warp] = last;
    __syncthreads();
    uint32_t pos = x - c, total = 0;
    int32_t row_last = -1;
    for (int w = 0; w < nwarp; ++w) {
        if (w < warp) pos += s_ws[w];
        total += s_ws[w];
        row_last = s_last[w] > row_last ? s_last[w] : row_last;
    }
    for (int j = 0; j < per; ++j) {
        const int i = tid * per + j;
        const int32_t e = i < p.K ? cand[i] : -1;
        if (e >= 0 && e < nb && pos < static_cast<uint32_t>(p.S)) sel[pos++] = e;
    }
    for (int i = static_cast<int>(total) + tid; i < p.S; i += T) sel[i] = -1;
    if (tid == 0) {
        const int32_t cnt = static_cast<int32_t>(total < static_cast<uint32_t>(p.S) ? total : p.S);
        int32_t fill = n - row_last * kBlock;
        fill = fill < 0 ? 0 : (fill > kBlock ? kBlock : fill);
        p.count[r] = cnt;
        p.end[r] = cnt > 0 ? (cnt - 1) * kBlock + fill : 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// finish: out[r, i] = the token of selected column cols[r, i] (a short row: the column; else sel[r, c / 8] * 8 + c % 8;
// -1 stays -1), and, with cand != nullptr, vLLM's candidate rule on the selected blocks: cand[r, j] = sel[r, j] unless
// it is -1, its score is -inf (NaN stays), or j >= kb; -1 then.

struct FinishParams {
    const int32_t* cols;
    const int32_t* sel;
    const int32_t* valid_blocks;  // with it: rows of at most short_blocks blocks have column = token
    const float* scores;
    int32_t* out;
    int32_t* cand;
    int64_t cols_stride, sel_stride, scores_stride, out_stride, cand_stride;
    int32_t topk, S, kb, cand_k, width_blocks, short_blocks;
};

__global__ void finish_kernel(const FinishParams p) {
    const int r = blockIdx.x;
    const int32_t* sel = p.sel + r * p.sel_stride;
    if (p.out != nullptr) {
        const int32_t* cols = p.cols + r * p.cols_stride;
        int32_t* out = p.out + r * p.out_stride;
        const bool identity = p.valid_blocks != nullptr && p.valid_blocks[r] <= p.short_blocks;
        for (int i = threadIdx.x; i < p.topk; i += blockDim.x) {
            const int32_t c = cols[i];
            int32_t tok = -1;
            if (identity) {
                tok = c;
            } else if (c >= 0 && (c >> 3) < p.S) {
                const int32_t b = sel[c >> 3];
                if (b >= 0) tok = b * kBlock + (c & 7);
            }
            out[i] = tok;
        }
    }
    if (p.cand != nullptr) {
        const float* scores = p.scores + r * p.scores_stride;
        int32_t* cand = p.cand + r * p.cand_stride;
        for (int j = threadIdx.x; j < p.cand_k; j += blockDim.x) {
            int32_t b = j < p.kb ? sel[j] : -1;
            if (b >= 0 && (b >= p.width_blocks || scores[b] == -INFINITY)) b = -1;
            cand[j] = b;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host

struct LaunchConfig {
    int threads;
    int ctas_per_sm;  // 0: as many as are resident at once
};

// Throughput grows with the warps resident per SM up to what the registers allow (measured 8 / 16 / 24 warps:
// 2.8 / 4.9 / 5.8 TB/s of K bytes): the scan 3 CTAs of 8 warps (24 warps), the rescore 11 CTAs of 2 warps (22).
LaunchConfig g_scan_cfg{256, 0}, g_rescore_cfg{64, 0};
// Tiles per chunk of a grouped scan (8 = 128 tokens): the warps' chunks together must stay well inside the L2
// (4,512 warps x 8 KiB = 35 MiB of 128), and each run of a chunk pays one un-overlapped fetch and a Q reload.
int g_scan_chunk = 8;
// A launch whose groups hold at most this many tiles (64 MiB of K, half the L2) is scanned row by row.
int g_scan_group_min_units = 65536;

int num_sms(int device) {
    static int cached[64] = {};
    if (device < 64 && cached[device]) return cached[device];
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    if (device < 64) cached[device] = sms;
    return sms;
}

// Dynamic shared memory past the 48 KiB default needs the kernel's opt-in (many rows: the scan's prefix sum).
template <typename Kernel>
void allow_smem(Kernel kernel, size_t smem, int device) {
    static size_t allowed[64] = {};
    const int d = device < 64 ? device : 63;
    if (smem <= (48u << 10) || smem <= allowed[d]) return;
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(smem <= static_cast<size_t>(optin), "block_topk: ", smem, " B of shared memory per CTA > ", optin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
    allowed[d] = static_cast<size_t>(optin);
}

template <typename Kernel>
int grid_x(Kernel kernel, const LaunchConfig& cfg, size_t smem, int device) {
    int per_sm = cfg.ctas_per_sm;
    if (per_sm <= 0) {
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, cfg.threads, smem));
        per_sm = per_sm < 1 ? 1 : per_sm;
    }
    return num_sms(device) * per_sm;
}

void check_q(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& w, int64_t rows) {
    TORCH_CHECK(q.is_cuda() && q.element_size() == 1 && q.dim() >= 3, "q: packed MXFP4 viewed as int8/uint8");
    TORCH_CHECK(q.size(-1) == kRowBytes && q.size(-2) == kHeads && q.stride(-1) == 1 && q.stride(-2) == kRowBytes,
                "q: [rows, (1,) 32, 64] with contiguous heads");
    TORCH_CHECK(q.size(0) == rows && q.numel() == rows * kHeads * kRowBytes, "q row count");
    TORCH_CHECK(sf_q.scalar_type() == torch::kInt32 && sf_q.size(-1) == kHeads && sf_q.stride(-1) == 1 &&
                    sf_q.size(0) == rows && sf_q.numel() == rows * kHeads,
                "q_scale: int32 [rows, (1,) 32] packed UE8M0");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.dim() == 2 && w.size(0) == rows && w.size(1) == kHeads &&
                    w.stride(1) == 1,
                "weights: fp32 [rows, 32] (the dense sm_120 path's weights)");
    TORCH_CHECK(q.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0, "q rows: 4-byte aligned");
}

void check_i32(const torch::Tensor& t, int64_t rows, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.is_contiguous() && t.numel() == rows, name,
                ": contiguous int32 [rows]");
}

void check_2d(const torch::Tensor& t, torch::ScalarType dtype, int64_t rows, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.dim() == 2 && t.size(0) >= rows && t.stride(1) == 1, name,
                ": [rows, n] with unit column stride");
}

void fill_q(Params& p, const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& w) {
    p.q = static_cast<const uint8_t*>(q.data_ptr());
    p.sf_q = static_cast<const uint32_t*>(sf_q.data_ptr());
    p.w = w.data_ptr<float>();
    p.q_stride = q.stride(0);
    p.sfq_stride = sf_q.stride(0);
    p.w_stride = w.stride(0);
}

void fill_paged(Params& p, const torch::Tensor& kv_cache, const torch::Tensor& lens, const torch::Tensor& bt,
                int64_t rows) {
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.element_size() == 1 && kv_cache.dim() == 4 && kv_cache.size(2) == 1 &&
                    kv_cache.size(3) == kRowBytes + 4,
                "kv_cache: uint8 [num_pages, page_kv, 1, 68] (MXFP4 indexer cache)");
    const int64_t page_kv = kv_cache.size(1);
    TORCH_CHECK(page_kv >= 16 && (page_kv & (page_kv - 1)) == 0, "page_kv: a power of two >= 16");
    TORCH_CHECK(kv_cache.stride(0) % 16 == 0 && kv_cache.stride(0) >= page_kv * (kRowBytes + 4) &&
                    reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % 16 == 0,
                "kv_cache: 16-byte aligned pages of page_kv x 64 B values then page_kv x 4 B scales");
    check_i32(lens, rows, "context_lens");
    check_2d(bt, torch::kInt32, rows, "block_table");
    p.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
    p.page_stride = kv_cache.stride(0);
    p.num_pages = kv_cache.size(0);
    p.page_shift = 0;
    while ((int64_t(1) << p.page_shift) < page_kv) ++p.page_shift;
    p.lens = lens.data_ptr<int32_t>();
    p.block_table = bt.data_ptr<int32_t>();
    p.bt_stride = bt.stride(0);
    p.bt_cols = bt.size(1);
}

void fill_contiguous(Params& p, const torch::Tensor& kv, const torch::Tensor& sf_kv, const torch::Tensor& ks,
                     const torch::Tensor& ke, int64_t rows) {
    TORCH_CHECK(kv.is_cuda() && kv.element_size() == 1 && kv.dim() == 2 && kv.size(1) == kRowBytes &&
                    kv.stride(0) == kRowBytes && kv.stride(1) == 1 &&
                    reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0 && kv.size(0) > 0,
                "kv: packed MXFP4 [num_kv, 64] contiguous, 16-byte aligned");
    TORCH_CHECK(sf_kv.is_cuda() && sf_kv.element_size() == 4 && sf_kv.numel() == kv.size(0) && sf_kv.is_contiguous(),
                "kv_scale: int32 [num_kv] packed UE8M0");
    check_i32(ks, rows, "cu_seqlen_ks");
    check_i32(ke, rows, "cu_seqlen_ke");
    p.kv = static_cast<const uint8_t*>(kv.data_ptr());
    p.sf_kv = static_cast<const uint32_t*>(sf_kv.data_ptr());
    p.num_kv = kv.size(0);
    p.ks = ks.data_ptr<int32_t>();
    p.ke = ke.data_ptr<int32_t>();
}

void fill_outputs(Params& p, const torch::Tensor* scores, const torch::Tensor& valid_blocks, const torch::Tensor& end,
                  const torch::Tensor& logits, int64_t rows, int64_t S, int64_t short_blocks, int64_t width_tokens,
                  bool always_scores = false) {
    check_2d(logits, torch::kFloat32, rows, "logits");
    check_i32(valid_blocks, rows, "valid_blocks");
    check_i32(end, rows, "end");
    TORCH_CHECK(S > 0 && width_tokens > 0 && width_tokens < (int64_t(1) << 30), "S / width_tokens out of range");
    TORCH_CHECK(short_blocks >= S, "short_blocks >= S (a long row has all S selected blocks)");
    const int64_t width_blocks = (width_tokens + kBlock - 1) / kBlock;
    const int64_t short_cols = (short_blocks < width_blocks ? short_blocks : width_blocks) * kBlock;
    TORCH_CHECK(logits.size(1) >= short_cols || logits.size(1) >= width_tokens,
                "logits: at least min(short_blocks * 8, width_tokens) columns");
    if (scores != nullptr) {
        check_2d(*scores, torch::kFloat32, rows, "scores");
        // unread when no row can be long and the scores are not asked for
        TORCH_CHECK((width_blocks <= short_blocks && !always_scores) || scores->size(1) >= width_blocks,
                    "scores: at least ceil(width_tokens / 8) columns");
        p.scores = scores->data_ptr<float>();
        p.scores_stride = scores->stride(0);
    }
    p.short_blocks = static_cast<int32_t>(short_blocks);
    p.logits = logits.data_ptr<float>();
    p.logits_stride = logits.stride(0);
    p.logits_width = static_cast<int32_t>(logits.size(1));
    p.valid_blocks = valid_blocks.data_ptr<int32_t>();
    p.end = end.data_ptr<int32_t>();
    p.rows = static_cast<int32_t>(rows);
    p.S = static_cast<int32_t>(S);
    p.width_tokens = static_cast<int32_t>(width_tokens);
    p.width_blocks = static_cast<int32_t>(width_blocks);
}

template <bool kPaged>
void launch_scan(const Params& p, int device, cudaStream_t stream) {
    const size_t smem = static_cast<size_t>(g_scan_cfg.threads >> 5) * kWarpSmem +
                        (static_cast<size_t>(p.rows) + 1 + 64) * sizeof(uint32_t);
    allow_smem(scan_kernel<kPaged>, smem, device);
    const int gx = grid_x(scan_kernel<kPaged>, g_scan_cfg, smem, device);
    scan_kernel<kPaged><<<gx, g_scan_cfg.threads, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kPaged>
void launch_rescore(const Params& p, int device, cudaStream_t stream) {
    const size_t smem = static_cast<size_t>(g_rescore_cfg.threads >> 5) * kWarpSmem +
                        (static_cast<size_t>(p.rows) + 1 + 64) * sizeof(uint32_t);
    allow_smem(rescore_kernel<kPaged>, smem, device);
    int gx = grid_x(rescore_kernel<kPaged>, g_rescore_cfg, smem, device);
    // no more warps than tiles
    const int64_t tiles = static_cast<int64_t>(p.rows) * ((p.S + 1) / 2);
    const int64_t need = (tiles + (g_rescore_cfg.threads >> 5) - 1) / (g_rescore_cfg.threads >> 5);
    gx = static_cast<int>(need < gx ? (need < 1 ? 1 : need) : gx);
    rescore_kernel<kPaged><<<gx, g_rescore_cfg.threads, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Row r = context positions [0, context_lens[r]) of a paged MXFP4 indexer cache (clamped to width_tokens).
// group: consecutive rows that read the same pages (1 + num_speculative_tokens; 1 = none; any value is correct).
// Writes valid_blocks[r] = ceil(len / 8), end[r] = len, and per row: more than S blocks -> scores[r, b] = the maximum of
// block b's logits (NaN-propagating; with pin_newest the newest block reads +inf) for b < valid_blocks[r]; at most S
// blocks -> logits[r, tok] for tok < len (and the scores as well with always_scores).
void paged_scan(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv_cache,
                const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& block_table,
                torch::Tensor scores, torch::Tensor valid_blocks, torch::Tensor sel_ends, torch::Tensor end,
                torch::Tensor logits, int64_t S, int64_t short_blocks, int64_t width_tokens, bool pin_newest,
                bool always_scores, int64_t group) {
    const int64_t rows = q.size(0);
    if (rows == 0) return;
    TORCH_CHECK(rows <= kMaxRows, "at most ", kMaxRows, " rows per launch");
    TORCH_CHECK(group >= 1 && group <= 64, "group: 1..64 rows");
    check_q(q, sf_q, weights, rows);
    Params p{};
    fill_q(p, q, sf_q, weights);
    fill_paged(p, kv_cache, context_lens, block_table, rows);
    fill_outputs(p, &scores, valid_blocks, end, logits, rows, S, short_blocks, width_tokens, always_scores);
    check_i32(sel_ends, rows, "sel_ends");
    p.sel_ends = sel_ends.data_ptr<int32_t>();
    p.pin = pin_newest;
    p.always_scores = always_scores;
    p.group = static_cast<int32_t>(group);
    p.chunk = g_scan_chunk;
    p.group_min_units = g_scan_group_min_units;
    const c10::cuda::CUDAGuard guard(q.device());
    launch_scan<true>(p, q.get_device(), c10::cuda::getCurrentCUDAStream());
}

// Row r = workspace tokens [cu_ks[r], cu_ke[r]) of the packed K workspace, positions relative to cu_ks[r].
void contiguous_scan(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv,
                     const torch::Tensor& sf_kv, const torch::Tensor& weights, const torch::Tensor& cu_ks,
                     const torch::Tensor& cu_ke, torch::Tensor scores, torch::Tensor valid_blocks,
                     torch::Tensor sel_ends, torch::Tensor end, torch::Tensor logits, int64_t S, int64_t short_blocks,
                     int64_t width_tokens, bool pin_newest, bool always_scores) {
    const int64_t rows = q.size(0);
    if (rows == 0) return;
    TORCH_CHECK(rows <= kMaxRows, "at most ", kMaxRows, " rows per launch");
    check_q(q, sf_q, weights, rows);
    Params p{};
    fill_q(p, q, sf_q, weights);
    fill_contiguous(p, kv, sf_kv, cu_ks, cu_ke, rows);
    fill_outputs(p, &scores, valid_blocks, end, logits, rows, S, short_blocks, width_tokens, always_scores);
    check_i32(sel_ends, rows, "sel_ends");
    p.sel_ends = sel_ends.data_ptr<int32_t>();
    p.pin = pin_newest;
    p.always_scores = always_scores;
    p.group = 1;  // a chunk's K workspace is far smaller than the L2
    p.chunk = g_scan_chunk;
    p.group_min_units = g_scan_group_min_units;
    const c10::cuda::CUDAGuard guard(q.device());
    launch_scan<false>(p, q.get_device(), c10::cuda::getCurrentCUDAStream());
}

// For rows of more than short_blocks blocks (valid_blocks from the scan): logits[r, j * 8 + o] = the logit of token
// sel[r, j] * 8 + o (sel: the S selected blocks, ascending), end[r] = the number of columns inside the row.
// by_count (a candidate consumer, after compact): every row's first valid_blocks[r] selected blocks; end is not written.
void paged_rescore(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv_cache,
                   const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& block_table,
                   const torch::Tensor& valid_blocks, const torch::Tensor& sel, torch::Tensor end, torch::Tensor logits,
                   int64_t short_blocks, int64_t width_tokens, bool by_count) {
    const int64_t rows = q.size(0);
    if (rows == 0) return;
    TORCH_CHECK(rows <= kMaxRows, "at most ", kMaxRows, " rows per launch");
    check_q(q, sf_q, weights, rows);
    check_2d(sel, torch::kInt32, rows, "sel");
    const int64_t S = sel.size(1);
    TORCH_CHECK(logits.size(1) >= S * kBlock, "logits: S * 8 columns");
    Params p{};
    fill_q(p, q, sf_q, weights);
    fill_paged(p, kv_cache, context_lens, block_table, rows);
    fill_outputs(p, nullptr, valid_blocks, end, logits, rows, S, by_count ? S : short_blocks, width_tokens);
    p.sel = sel.data_ptr<int32_t>();
    p.sel_stride = sel.stride(0);
    p.by_count = by_count;
    const c10::cuda::CUDAGuard guard(q.device());
    launch_rescore<true>(p, q.get_device(), c10::cuda::getCurrentCUDAStream());
}

void contiguous_rescore(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv,
                        const torch::Tensor& sf_kv, const torch::Tensor& weights, const torch::Tensor& cu_ks,
                        const torch::Tensor& cu_ke, const torch::Tensor& valid_blocks, const torch::Tensor& sel,
                        torch::Tensor end, torch::Tensor logits, int64_t short_blocks, int64_t width_tokens,
                        bool by_count) {
    const int64_t rows = q.size(0);
    if (rows == 0) return;
    TORCH_CHECK(rows <= kMaxRows, "at most ", kMaxRows, " rows per launch");
    check_q(q, sf_q, weights, rows);
    check_2d(sel, torch::kInt32, rows, "sel");
    const int64_t S = sel.size(1);
    TORCH_CHECK(logits.size(1) >= S * kBlock, "logits: S * 8 columns");
    Params p{};
    fill_q(p, q, sf_q, weights);
    fill_contiguous(p, kv, sf_kv, cu_ks, cu_ke, rows);
    fill_outputs(p, nullptr, valid_blocks, end, logits, rows, S, by_count ? S : short_blocks, width_tokens);
    p.sel = sel.data_ptr<int32_t>();
    p.sel_stride = sel.stride(0);
    p.by_count = by_count;
    const c10::cuda::CUDAGuard guard(q.device());
    launch_rescore<false>(p, q.get_device(), c10::cuda::getCurrentCUDAStream());
}

// A candidate consumer's rows: sel / count / end from its candidate blocks (see compact_kernel). lens: the paged
// rows' context lengths; with cu_ke given, the rows are [lens[r], cu_ke[r]) of the packed workspace.
void compact(const torch::Tensor& cand, const torch::Tensor& lens, const c10::optional<torch::Tensor>& cu_ke,
             torch::Tensor sel, torch::Tensor count, torch::Tensor end, int64_t width_tokens, int64_t rows) {
    if (rows == 0) return;
    check_2d(cand, torch::kInt32, rows, "candidates");
    check_2d(sel, torch::kInt32, rows, "sel");
    check_i32(lens, rows, "context_lens / cu_seqlen_ks");
    check_i32(count, rows, "count");
    check_i32(end, rows, "end");
    TORCH_CHECK(width_tokens > 0 && width_tokens < (int64_t(1) << 30), "width_tokens out of range");
    CompactParams p{};
    p.cand = cand.data_ptr<int32_t>();
    p.cand_stride = cand.stride(0);
    p.K = static_cast<int32_t>(cand.size(1));
    p.sel = sel.data_ptr<int32_t>();
    p.sel_stride = sel.stride(0);
    p.S = static_cast<int32_t>(sel.size(1));
    p.count = count.data_ptr<int32_t>();
    p.end = end.data_ptr<int32_t>();
    p.width_tokens = static_cast<int32_t>(width_tokens);
    const c10::cuda::CUDAGuard guard(cand.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (cu_ke.has_value()) {
        check_i32(*cu_ke, rows, "cu_seqlen_ke");
        p.ks = lens.data_ptr<int32_t>();
        p.ke = cu_ke->data_ptr<int32_t>();
        compact_kernel<false><<<static_cast<unsigned>(rows), 256, 0, stream>>>(p);
    } else {
        p.lens = lens.data_ptr<int32_t>();
        compact_kernel<true><<<static_cast<unsigned>(rows), 256, 0, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// valid_blocks (None: no short rows): rows of at most short_blocks blocks have column = token.
// out[r, :topk] = the tokens of the selected columns cols[r, :topk] (None: skipped); cand[r, :] (None: skipped) = the
// selected blocks sel[r, :kb] by vLLM's candidate rule.
void finish(const c10::optional<torch::Tensor>& cols, const torch::Tensor& sel,
            const c10::optional<torch::Tensor>& out, int64_t topk, const c10::optional<torch::Tensor>& scores,
            const c10::optional<torch::Tensor>& cand, int64_t kb, const c10::optional<torch::Tensor>& valid_blocks,
            int64_t short_blocks, int64_t rows) {
    if (rows == 0) return;
    check_2d(sel, torch::kInt32, rows, "sel");
    FinishParams p{};
    p.sel = sel.data_ptr<int32_t>();
    p.sel_stride = sel.stride(0);
    p.S = static_cast<int32_t>(sel.size(1));
    p.topk = static_cast<int32_t>(topk);
    if (valid_blocks.has_value()) {
        check_i32(*valid_blocks, rows, "valid_blocks");
        p.valid_blocks = valid_blocks->data_ptr<int32_t>();
        p.short_blocks = static_cast<int32_t>(short_blocks);
    }
    if (out.has_value()) {
        TORCH_CHECK(cols.has_value(), "cols with out");
        check_2d(*cols, torch::kInt32, rows, "cols");
        check_2d(*out, torch::kInt32, rows, "out");
        TORCH_CHECK(cols->size(1) >= topk && out->size(1) >= topk, "cols / out: topk columns");
        p.cols = cols->data_ptr<int32_t>();
        p.cols_stride = cols->stride(0);
        p.out = out->data_ptr<int32_t>();
        p.out_stride = out->stride(0);
    }
    if (cand.has_value()) {
        TORCH_CHECK(scores.has_value(), "scores with cand");
        check_2d(*scores, torch::kFloat32, rows, "scores");
        check_2d(*cand, torch::kInt32, rows, "cand");
        TORCH_CHECK(kb >= 0 && kb <= sel.size(1), "kb <= S");
        p.scores = scores->data_ptr<float>();
        p.scores_stride = scores->stride(0);
        p.width_blocks = static_cast<int32_t>(scores->size(1));
        p.cand = cand->data_ptr<int32_t>();
        p.cand_stride = cand->stride(0);
        p.cand_k = static_cast<int32_t>(cand->size(1));
        p.kb = static_cast<int32_t>(kb);
    }
    if (p.out == nullptr && p.cand == nullptr) return;
    const c10::cuda::CUDAGuard guard(sel.device());
    finish_kernel<<<static_cast<unsigned>(rows), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch shapes (tuning): threads per CTA (a multiple of 32), CTAs per SM (0 = as many as are resident), and the
// grouped scan's tiles per chunk.
void set_launch(int64_t scan_threads, int64_t scan_ctas_per_sm, int64_t rescore_threads, int64_t rescore_ctas_per_sm,
                int64_t scan_chunk) {
    TORCH_CHECK(scan_threads % 32 == 0 && scan_threads >= 32 && scan_threads <= 256 && rescore_threads % 32 == 0 &&
                    rescore_threads >= 32 && rescore_threads <= 256,
                "threads: a multiple of 32, at most 256");
    g_scan_cfg.threads = static_cast<int>(scan_threads);
    g_scan_cfg.ctas_per_sm = static_cast<int>(scan_ctas_per_sm);
    g_rescore_cfg.threads = static_cast<int>(rescore_threads);
    g_rescore_cfg.ctas_per_sm = static_cast<int>(rescore_ctas_per_sm);
    TORCH_CHECK(scan_chunk >= 1, "scan_chunk >= 1 tiles");
    g_scan_chunk = static_cast<int>(scan_chunk);
}

int64_t max_rows() { return kMaxRows; }

// Tuning / tests: the grouped scan's threshold in tiles (0: always grouped when group > 1).
void set_scan_group_min_units(int64_t units) { g_scan_group_min_units = static_cast<int>(units); }

// [registers per thread, CTAs per SM at the configured launch shape] of the paged scan and rescore kernels.
std::vector<int64_t> kernel_info(int64_t device) {
    cudaFuncAttributes fs, fr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fs, scan_kernel<true>));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fr, rescore_kernel<true>));
    const int d = static_cast<int>(device);
    const size_t smem_s = static_cast<size_t>(g_scan_cfg.threads >> 5) * kWarpSmem + 1024 * sizeof(uint32_t);
    const size_t smem_r = static_cast<size_t>(g_rescore_cfg.threads >> 5) * kWarpSmem + 1024 * sizeof(uint32_t);
    return {fs.numRegs, grid_x(scan_kernel<true>, g_scan_cfg, smem_s, d) / num_sms(d), fr.numRegs,
            grid_x(rescore_kernel<true>, g_rescore_cfg, smem_r, d) / num_sms(d)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_scan", &paged_scan);
    m.def("contiguous_scan", &contiguous_scan);
    m.def("paged_rescore", &paged_rescore);
    m.def("contiguous_rescore", &contiguous_rescore);
    m.def("compact", &compact);
    m.def("finish", &finish);
    m.def("set_launch", &set_launch);
    m.def("max_rows", &max_rows);
    m.def("set_scan_group_min_units", &set_scan_group_min_units);
    m.def("kernel_info", &kernel_info);
}
