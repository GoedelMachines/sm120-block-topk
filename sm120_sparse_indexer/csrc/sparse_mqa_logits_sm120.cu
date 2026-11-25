// Sparse MQA logits for the DeepSeek-V4.1 two-level indexer on sm_120 (RTX PRO 6000 / RTX 5090).
//
// What it is: DeepGEMM's sparse MQA-logits API (fp8_fp4_(paged_)sparse_mqa_logits + their metadata) for the MXFP4
// indexer cache, computing at every candidate token exactly the fp32 value DeepGEMM's sm_120 *dense* kernels
// (sm120_fp4_paged_mqa_logits / sm120_fp4_mqa_logits) compute there:
//
//   logit(q, tok) = sum over the 4 lanes t of a quad of P_t,   P_t = (((0 + T_0) + T_1) + T_2) + T_3,
//   T_nt = fma(relu(d[h0]), w[h0], relu(d[h0 + 1]) * w[h0 + 1]),  h0 = nt * 8 + 2t,
//   d[h] = the m16n8k64 MXF4 block-scaled MMA of the token's K row (A) against head h's Q row (B), two k-steps of 64,
//   reduced over the quad by two xor-shuffles ((P_t + P_t^1) + (P_t^2 + P_t^3)).
//
// The dense kernels' SASS (nvcc 13.0 -O3, sm_120f) is exactly this sequence (FMNMX, FMUL, FFMA, FADD RZ, FADD x3,
// SHFL.BFLY 1/2 + FADD); here every step is spelled with an _rn intrinsic so no compiler contraction can change it.
// A token's MMA row does not depend on the other rows of the tile, so gathering 8- or 16-token candidate blocks into
// the A tile in any order leaves each value unchanged. The output is fp32 (DeepGEMM's SM100 sparse kernels return bf16;
// the dense sm_120 path ranks fp32 logits, and bf16 would round ties into the top-k).
//
// Metadata (int32 [rows, S + 2], S = num_max_sparse_blocks), built on the device, graph-safe:
//   paged:      entry[r, j] = (physical sparse block << 5) | n_valid (tokens [0, n_valid) of the block are < context),
//               physical sparse block = page * (page_kv / SBK) + (logical token % page_kv) / SBK; -1 = skip.
//   contiguous: entry[r, j] = first workspace token of the block (si * SBK + (unaligned ? ks % SBK : 0)), -1 = skip;
//               entry[r, S] / entry[r, S + 1] = the row's valid token range [max(ks, 0), min(ke, num_kv)).
//   A block repeating the previous one (DeepGEMM's padding convention) is skipped: its columns read -inf.
// Logits: fp32 [rows, S * SBK]; column j * SBK + o scores token o of block j, or -inf where that token is outside the
// row's range or the block is skipped.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kHeads = 32;     // index_n_heads
constexpr int kRowBytes = 64;  // MXFP4 head_dim 128: 64 bytes of e2m1 pairs (+ 4 UE8M0 scale bytes elsewhere)
constexpr int kThreads = 128;  // 4 warps

// ---------------------------------------------------------------------------------------------------------------
// The pieces of DeepGEMM's sm_120 kernels the arithmetic depends on (deep_gemm/mma/sm120.cuh,
// deep_gemm/common/sm120_utils.cuh): the MXF4 block-scaled MMA, the 64-byte swizzle and the ldmatrix fragment loads.

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

__device__ __forceinline__ uint16_t extract_sf_pair(uint32_t packed, uint32_t first_byte_idx) {
    return static_cast<uint16_t>((packed >> (first_byte_idx * 8)) & 0xFFFF);
}

// CuTe Swizzle<2, 4, 3> (TMA SWIZZLE_64B): bits [4:5] ^= bits [7:8]; the tiles are 1024-byte aligned.
__device__ __forceinline__ int swz64(int x) { return x ^ ((x & 0x180) >> 3); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&d)[4], const void* smem) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "l"(__cvta_generic_to_shared(smem))
                 : "memory");
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& d0, uint32_t& d1, const void* smem) {
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "l"(__cvta_generic_to_shared(smem))
                 : "memory");
}

// load_a_fragment<64>: lane row (lane & 7) + ((lane >> 3) & 1) * 8 of the 16-row tile, column (lane >> 4) * 16 + k * 32
__device__ __forceinline__ void load_a(uint32_t (&frag)[4], const uint8_t* tile, int lane, int k) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int base = row * kRowBytes;
    const int xr = swz64(base) ^ base;
    ldmatrix_x4(frag, tile + base + ((((lane >> 4) * 16) + k * 32) ^ xr));
}

// load_b_fragment_x2<64>: lane row (lane & 7) + n_tile * 8, column ((lane >> 3) & 1) * 16 + k * 32
__device__ __forceinline__ void load_b(uint32_t& b0, uint32_t& b1, const uint8_t* q, int lane, int n_tile, int k) {
    const int row = (lane & 7) + n_tile * 8;
    const int base = row * kRowBytes;
    const int xr = swz64(base) ^ base;
    ldmatrix_x2(b0, b1, q + base + (((((lane >> 3) & 1) * 16) + k * 32) ^ xr));
}

// ---------------------------------------------------------------------------------------------------------------
// cp.async with zero fill

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(src), "r"(valid ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(addr), "l"(src), "r"(valid ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// ---------------------------------------------------------------------------------------------------------------
// Metadata

template <int SBK>
__global__ void __launch_bounds__(256) paged_metadata_kernel(const int32_t* __restrict__ context_lens,
                                                             const int32_t* __restrict__ block_table,
                                                             int64_t bt_stride, int64_t num_bt_cols,
                                                             const int32_t* __restrict__ sparse_idx,
                                                             int64_t si_stride, int num_blocks, int page_kv,
                                                             int64_t num_pages, int32_t* __restrict__ meta) {
    const int row = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t mstride = num_blocks + 2;
    if (j >= num_blocks) {
        if (j < num_blocks + 2) meta[row * mstride + j] = 0;
        return;
    }
    const int32_t* si = sparse_idx + row * si_stride;
    const int32_t b = si[j];
    const int32_t prev = j > 0 ? si[j - 1] : -1;
    const int64_t ctx = context_lens[row];
    const int64_t start = static_cast<int64_t>(b) * SBK;
    int32_t entry = -1;
    if (b >= 0 && b != prev && start < ctx) {
        const int64_t lp = start / page_kv;
        if (lp < num_bt_cols) {
            const int64_t page = block_table[row * bt_stride + lp];
            if (page >= 0 && page < num_pages) {
                const int64_t phys = page * (page_kv / SBK) + (start % page_kv) / SBK;
                const int n = static_cast<int>(ctx - start < SBK ? ctx - start : SBK);
                entry = static_cast<int32_t>((phys << 5) | n);
            }
        }
    }
    meta[row * mstride + j] = entry;
}

template <int SBK>
__global__ void __launch_bounds__(256) contiguous_metadata_kernel(const int32_t* __restrict__ ks_ptr,
                                                                  const int32_t* __restrict__ ke_ptr,
                                                                  int64_t num_kv, const int32_t* __restrict__ sparse_idx,
                                                                  int64_t si_stride, int num_blocks, bool unaligned,
                                                                  int32_t* __restrict__ meta) {
    const int row = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t mstride = num_blocks + 2;
    const int64_t ks = ks_ptr[row], ke = ke_ptr[row];
    const int64_t lo = ks > 0 ? ks : 0;
    const int64_t hi = ke < num_kv ? ke : num_kv;
    if (j >= num_blocks) {
        if (j == num_blocks) meta[row * mstride + j] = static_cast<int32_t>(lo);
        if (j == num_blocks + 1) meta[row * mstride + j] = static_cast<int32_t>(hi);
        return;
    }
    const int32_t* si = sparse_idx + row * si_stride;
    const int32_t b = si[j];
    const int32_t prev = j > 0 ? si[j - 1] : -1;
    const int64_t off = unaligned ? ((ks % SBK) + SBK) % SBK : 0;
    const int64_t start = static_cast<int64_t>(b) * SBK + off;
    int32_t entry = -1;
    if (b >= 0 && b != prev && start < hi && start + SBK > lo) entry = static_cast<int32_t>(start);
    meta[row * mstride + j] = entry;
}

// ---------------------------------------------------------------------------------------------------------------
// Logits: one CTA of 4 warps per (row, 128 consecutive sparse columns); each warp scores two 16-token MMA tiles.

constexpr int kTokensPerCTA = 128;

struct LogitsParams {
    const uint8_t* q;         // [rows, 32, 64] packed e2m1 (row stride q_stride bytes)
    const uint8_t* sf_q;      // [rows, 32] int32 packed UE8M0 (row stride sfq_stride bytes)
    const void* weights;      // [rows, 32] fp32 or bf16 (row stride w_stride elements)
    const uint8_t* kv;        // paged: page base; contiguous: [num_kv, 64]
    const uint8_t* sf_kv;     // contiguous only: [num_kv] int32
    const int32_t* meta;      // [rows, S + 2]
    float* logits;            // [rows, S * SBK]
    int64_t q_stride, sfq_stride, w_stride, page_stride, logits_stride;
    int64_t num_pages;        // paged: pages in the cache (entries past it read as invalid)
    int num_blocks;           // S
    int page_kv;
};

// Whether token o of the block with metadata entry e is scored; paged: also its page index.
template <int SBK, bool kPaged>
__device__ __forceinline__ bool token_valid(const LogitsParams& p, int32_t e, int o, int64_t lo, int64_t hi) {
    if (e < 0) return false;
    if constexpr (kPaged) {
        return o < (e & 31) && ((e >> 5) / (p.page_kv / SBK)) < p.num_pages;
    } else {
        const int64_t t = static_cast<int64_t>(e) + o;
        return t >= lo && t < hi;
    }
}

template <int SBK, bool kPaged, bool kWeightsFP32>
__global__ void __launch_bounds__(kThreads) sparse_logits_kernel(const LogitsParams p) {
    constexpr int kBlocks = kTokensPerCTA / SBK;
    __shared__ __align__(1024) uint8_t s_kv[kTokensPerCTA * kRowBytes];
    __shared__ __align__(1024) uint8_t s_q[kHeads * kRowBytes];
    __shared__ __align__(16) uint32_t s_sfkv[kTokensPerCTA];
    __shared__ __align__(16) uint32_t s_sfq[kHeads];
    __shared__ __align__(16) float s_w[kHeads];
    __shared__ int32_t s_ent[kBlocks];

    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int j0 = blockIdx.x * kBlocks;
    const int32_t* meta = p.meta + static_cast<int64_t>(row) * (p.num_blocks + 2);
    float* out = p.logits + static_cast<int64_t>(row) * p.logits_stride + static_cast<int64_t>(j0) * SBK;
    const int ncols = (p.num_blocks - j0 < kBlocks ? p.num_blocks - j0 : kBlocks) * SBK;

    int64_t lo = 0, hi = 0;
    if constexpr (!kPaged) {
        lo = meta[p.num_blocks];
        hi = meta[p.num_blocks + 1];
    }
    bool mine = false;
    if (tid < kBlocks) {
        const int32_t e = j0 + tid < p.num_blocks ? meta[j0 + tid] : -1;
        s_ent[tid] = e;
        mine = e >= 0;
    }
    if (!__syncthreads_or(mine)) {
        for (int c = tid; c < ncols; c += kThreads) out[c] = -INFINITY;
        return;
    }

    // Q, its scales and the weights of this row
    {
        const uint8_t* q = p.q + row * p.q_stride;
        const int r = tid >> 2, ch = tid & 3;  // 128 chunks of 16 B: head r, chunk ch
        cp_async16(s_q + swz64(r * kRowBytes + ch * 16), q + r * kRowBytes + ch * 16, true);
        if (tid < 8) cp_async16(s_sfq + tid * 4, p.sf_q + row * p.sfq_stride + tid * 16, true);
        if constexpr (kWeightsFP32) {
            if (tid >= 8 && tid < 16)
                cp_async16(s_w + (tid - 8) * 4,
                           static_cast<const float*>(p.weights) + row * p.w_stride + (tid - 8) * 4, true);
        } else {
            if (tid >= 8 && tid < 8 + kHeads)
                s_w[tid - 8] = __bfloat162float(static_cast<const __nv_bfloat16*>(p.weights)[row * p.w_stride + tid - 8]);
        }
    }
    // KV rows: 128 tokens x 4 chunks of 16 B, and the 128 scales
    #pragma unroll
    for (int i = 0; i < kTokensPerCTA * 4 / kThreads; ++i) {
        const int c = tid + i * kThreads;
        const int tok = c >> 2, ch = c & 3;
        const int blk = tok / SBK, o = tok % SBK;
        const int32_t e = s_ent[blk];
        const uint8_t* src = p.kv;
        const bool valid = token_valid<SBK, kPaged>(p, e, o, lo, hi);
        if (valid) {
            if constexpr (kPaged) {
                const int64_t phys = e >> 5;
                const int bpp = p.page_kv / SBK;
                src = p.kv + (phys / bpp) * p.page_stride + ((phys % bpp) * SBK + o) * kRowBytes + ch * 16;
            } else {
                src = p.kv + (static_cast<int64_t>(e) + o) * kRowBytes + ch * 16;
            }
        }
        cp_async16(s_kv + swz64(tok * kRowBytes + ch * 16), src, valid);
    }
    {
        const int tok = tid;
        const int blk = tok / SBK, o = tok % SBK;
        const int32_t e = s_ent[blk];
        const uint8_t* src = p.kv;
        const bool valid = token_valid<SBK, kPaged>(p, e, o, lo, hi);
        if (valid) {
            if constexpr (kPaged) {
                const int64_t phys = e >> 5;
                const int bpp = p.page_kv / SBK;
                src = p.kv + (phys / bpp) * p.page_stride + static_cast<int64_t>(p.page_kv) * kRowBytes +
                      ((phys % bpp) * SBK + o) * 4;
            } else {
                src = p.sf_kv + (static_cast<int64_t>(e) + o) * 4;
            }
        }
        cp_async4(s_sfkv + tok, src, valid);
    }
    cp_async_wait_all();
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;

    // This warp's B operands (Q rows of heads nt * 8 .. nt * 8 + 7), Q scales and weights, for all four n-tiles.
    uint32_t b[4][2][2];
    uint32_t sfq[4];
    float w0[4], w1[4];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) load_b(b[nt][k][0], b[nt][k][1], s_q, lane, nt, k);
        sfq[nt] = s_sfq[nt * 8 + g];
        w0[nt] = s_w[nt * 8 + t * 2];
        w1[nt] = s_w[nt * 8 + t * 2 + 1];
    }

    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int row0 = warp * 32 + tile * 16;
        const uint8_t* a_tile = s_kv + row0 * kRowBytes;  // 1024-byte aligned: the swizzle phase is unchanged
        const uint32_t sfa = s_sfkv[row0 + g + (t & 1) * 8];
        uint32_t a[2][4];
        load_a(a[0], a_tile, lane, 0);
        load_a(a[1], a_tile, lane, 1);

        float p0 = 0.0f, p1 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int k = 0; k < 2; ++k)
                fp4_mma_block_scaled(d, a[k], b[nt][k][0], b[nt][k][1], extract_sf_pair(sfa, k * 2),
                                     extract_sf_pair(sfq[nt], k * 2));
            const float t0 = __fmaf_rn(fmaxf(d[0], 0.0f), w0[nt], __fmul_rn(fmaxf(d[1], 0.0f), w1[nt]));
            const float t1 = __fmaf_rn(fmaxf(d[2], 0.0f), w0[nt], __fmul_rn(fmaxf(d[3], 0.0f), w1[nt]));
            p0 = __fadd_rn(p0, t0);
            p1 = __fadd_rn(p1, t1);
        }
        #pragma unroll
        for (int s = 1; s <= 2; s <<= 1) {
            p0 = __fadd_rn(p0, __shfl_xor_sync(0xffffffffu, p0, s));
            p1 = __fadd_rn(p1, __shfl_xor_sync(0xffffffffu, p1, s));
        }
        if (t < 2) {
            const int tok = row0 + g + t * 8;
            const float v = t == 0 ? p0 : p1;
            const int blk = tok / SBK, o = tok % SBK;
            const bool valid = token_valid<SBK, kPaged>(p, s_ent[blk], o, lo, hi);
            if (tok < ncols) out[tok] = valid ? v : -INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host

void check_sbk(int64_t sbk) { TORCH_CHECK(sbk == 8 || sbk == 16, "sparse_block_kv must be 8 or 16"); }

void check_int32_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 2 && t.stride(1) == 1, name,
                ": int32 [rows, n] with unit column stride");
}

}  // namespace

torch::Tensor paged_metadata(const torch::Tensor& context_lens, const torch::Tensor& block_table,
                             const torch::Tensor& sparse_indices, int64_t page_kv, int64_t sbk, int64_t num_pages) {
    check_sbk(sbk);
    check_int32_2d(sparse_indices, "sparse_kv_block_indices");
    check_int32_2d(block_table, "block_table");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt32 && context_lens.is_contiguous(), "context_lens: int32");
    const int64_t rows = sparse_indices.size(0), S = sparse_indices.size(1);
    TORCH_CHECK(context_lens.numel() == rows && block_table.size(0) >= rows, "row count mismatch");
    TORCH_CHECK(page_kv > 0 && page_kv % sbk == 0, "page_kv must be a multiple of sparse_block_kv");
    TORCH_CHECK(num_pages * (page_kv / sbk) < (int64_t(1) << 26), "too many pages for the packed metadata");
    auto meta = torch::empty({rows, S + 2}, sparse_indices.options());
    if (rows == 0) return meta;
    const c10::cuda::CUDAGuard guard(sparse_indices.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 grid((S + 2 + 255) / 256, rows);
    auto kernel = sbk == 8 ? paged_metadata_kernel<8> : paged_metadata_kernel<16>;
    kernel<<<grid, 256, 0, stream>>>(context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                                     block_table.stride(0), block_table.size(1), sparse_indices.data_ptr<int32_t>(),
                                     sparse_indices.stride(0), static_cast<int>(S), static_cast<int>(page_kv),
                                     num_pages, meta.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return meta;
}

torch::Tensor contiguous_metadata(const torch::Tensor& cu_ks, const torch::Tensor& cu_ke, int64_t num_kv,
                                  const torch::Tensor& sparse_indices, int64_t sbk, bool unaligned) {
    check_sbk(sbk);
    check_int32_2d(sparse_indices, "sparse_kv_block_indices");
    TORCH_CHECK(cu_ks.scalar_type() == torch::kInt32 && cu_ke.scalar_type() == torch::kInt32 && cu_ks.is_contiguous() &&
                    cu_ke.is_contiguous(),
                "cu_seqlen_ks / cu_seqlen_ke: contiguous int32");
    const int64_t rows = sparse_indices.size(0), S = sparse_indices.size(1);
    TORCH_CHECK(cu_ks.numel() == rows && cu_ke.numel() == rows, "row count mismatch");
    TORCH_CHECK(num_kv < (int64_t(1) << 31) - 16, "num_kv_tokens out of int32 range");
    auto meta = torch::empty({rows, S + 2}, sparse_indices.options());
    if (rows == 0) return meta;
    const c10::cuda::CUDAGuard guard(sparse_indices.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 grid((S + 2 + 255) / 256, rows);
    auto kernel = sbk == 8 ? contiguous_metadata_kernel<8> : contiguous_metadata_kernel<16>;
    kernel<<<grid, 256, 0, stream>>>(cu_ks.data_ptr<int32_t>(), cu_ke.data_ptr<int32_t>(), num_kv,
                                     sparse_indices.data_ptr<int32_t>(), sparse_indices.stride(0), static_cast<int>(S),
                                     unaligned, meta.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return meta;
}

namespace {

template <bool kPaged>
void launch_logits(LogitsParams& p, int64_t rows, int64_t sbk, bool w32, cudaStream_t stream) {
    const int kBlocks = static_cast<int>(kTokensPerCTA / sbk);
    dim3 grid((p.num_blocks + kBlocks - 1) / kBlocks, static_cast<unsigned>(rows));
    if (sbk == 8) {
        if (w32) sparse_logits_kernel<8, kPaged, true><<<grid, kThreads, 0, stream>>>(p);
        else sparse_logits_kernel<8, kPaged, false><<<grid, kThreads, 0, stream>>>(p);
    } else {
        if (w32) sparse_logits_kernel<16, kPaged, true><<<grid, kThreads, 0, stream>>>(p);
        else sparse_logits_kernel<16, kPaged, false><<<grid, kThreads, 0, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_q(const torch::Tensor& q, const torch::Tensor& sf_q, int64_t rows) {
    TORCH_CHECK(q.is_cuda() && q.element_size() == 1, "q: packed MXFP4 viewed as int8/uint8");
    TORCH_CHECK(q.size(-1) == kRowBytes && q.size(-2) == kHeads && q.stride(-1) == 1 && q.stride(-2) == kRowBytes,
                "q: [rows, (1,) 32, 64] with contiguous heads");
    TORCH_CHECK(q.numel() / (kHeads * kRowBytes) == rows, "q row count");
    TORCH_CHECK(sf_q.scalar_type() == torch::kInt32 && sf_q.size(-1) == kHeads && sf_q.stride(-1) == 1 &&
                    sf_q.numel() / kHeads == rows,
                "q_scale: int32 [rows, (1,) 32] packed UE8M0");
    TORCH_CHECK((q.stride(0) * 1) % 16 == 0 && (sf_q.stride(0) * 4) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(sf_q.data_ptr()) % 16 == 0,
                "q / q_scale rows must be 16-byte aligned");
}

void fill_q(LogitsParams& p, const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& weights,
            int64_t rows) {
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == kHeads && weights.stride(1) == 1,
                "weights: [rows, 32]");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 || weights.scalar_type() == torch::kBFloat16,
                "weights: fp32 (exact vs the dense sm_120 path) or bf16");
    if (weights.scalar_type() == torch::kFloat32)
        TORCH_CHECK(weights.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0,
                    "fp32 weights rows must be 16-byte aligned");
    p.q = static_cast<const uint8_t*>(q.data_ptr());
    p.sf_q = static_cast<const uint8_t*>(sf_q.data_ptr());
    p.weights = weights.data_ptr();
    p.q_stride = q.stride(0);
    p.sfq_stride = sf_q.stride(0) * 4;
    p.w_stride = weights.stride(0);
}

}  // namespace

torch::Tensor paged_sparse_logits(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv_cache,
                                  const torch::Tensor& weights, const torch::Tensor& meta, int64_t num_blocks,
                                  int64_t sbk) {
    check_sbk(sbk);
    const int64_t rows = q.size(0);
    check_q(q, sf_q, rows);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.element_size() == 1 && kv_cache.dim() == 4 && kv_cache.size(2) == 1 &&
                    kv_cache.size(3) == kRowBytes + 4,
                "kv_cache: uint8 [num_pages, page_kv, 1, 68] (MXFP4 indexer cache)");
    const int64_t page_kv = kv_cache.size(1);
    TORCH_CHECK(page_kv % sbk == 0 && kv_cache.stride(0) % 256 == 0 && kv_cache.stride(0) >= page_kv * (kRowBytes + 4),
                "kv_cache: page_kv % sparse_block_kv == 0 and a 256-byte aligned page stride");
    check_int32_2d(meta, "metadata");
    TORCH_CHECK(meta.size(0) == rows && meta.size(1) == num_blocks + 2 && meta.is_contiguous(),
                "metadata from get_paged_sparse_mqa_logits_metadata for these rows");
    auto logits = torch::empty({rows, num_blocks * sbk}, q.options().dtype(torch::kFloat32));
    if (rows == 0 || num_blocks == 0) return logits;
    LogitsParams p{};
    fill_q(p, q, sf_q, weights, rows);
    p.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
    p.sf_kv = nullptr;
    p.meta = meta.data_ptr<int32_t>();
    p.logits = logits.data_ptr<float>();
    p.page_stride = kv_cache.stride(0);
    p.num_pages = kv_cache.size(0);
    p.logits_stride = logits.stride(0);
    p.num_blocks = static_cast<int>(num_blocks);
    p.page_kv = static_cast<int>(page_kv);
    const c10::cuda::CUDAGuard guard(q.device());
    launch_logits<true>(p, rows, sbk, weights.scalar_type() == torch::kFloat32, c10::cuda::getCurrentCUDAStream());
    return logits;
}

torch::Tensor contiguous_sparse_logits(const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv,
                                       const torch::Tensor& sf_kv, const torch::Tensor& weights,
                                       const torch::Tensor& meta, int64_t num_blocks, int64_t sbk) {
    check_sbk(sbk);
    const int64_t rows = q.size(0);
    check_q(q, sf_q, rows);
    TORCH_CHECK(kv.is_cuda() && kv.element_size() == 1 && kv.dim() == 2 && kv.size(1) == kRowBytes &&
                    kv.stride(0) == kRowBytes && kv.stride(1) == 1,
                "kv: packed MXFP4 [num_kv, 64] contiguous");
    TORCH_CHECK(sf_kv.element_size() == 4 && sf_kv.numel() == kv.size(0) && sf_kv.is_contiguous(),
                "kv_scale: int32 [num_kv] packed UE8M0");
    check_int32_2d(meta, "metadata");
    TORCH_CHECK(meta.size(0) == rows && meta.size(1) == num_blocks + 2 && meta.is_contiguous(),
                "metadata from get_sparse_mqa_logits_metadata for these rows");
    auto logits = torch::empty({rows, num_blocks * sbk}, q.options().dtype(torch::kFloat32));
    if (rows == 0 || num_blocks == 0) return logits;
    LogitsParams p{};
    fill_q(p, q, sf_q, weights, rows);
    p.kv = static_cast<const uint8_t*>(kv.data_ptr());
    p.sf_kv = static_cast<const uint8_t*>(sf_kv.data_ptr());
    p.meta = meta.data_ptr<int32_t>();
    p.logits = logits.data_ptr<float>();
    p.page_stride = 0;
    p.num_pages = 0;
    p.logits_stride = logits.stride(0);
    p.num_blocks = static_cast<int>(num_blocks);
    p.page_kv = 0;
    const c10::cuda::CUDAGuard guard(q.device());
    launch_logits<false>(p, rows, sbk, weights.scalar_type() == torch::kFloat32, c10::cuda::getCurrentCUDAStream());
    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_metadata", &paged_metadata);
    m.def("contiguous_metadata", &contiguous_metadata);
    m.def("paged_sparse_logits", &paged_sparse_logits);
    m.def("contiguous_sparse_logits", &contiguous_sparse_logits);
}
