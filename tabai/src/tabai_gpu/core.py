from contextvars import ContextVar

import cupy as cp
import numpy as np

_BLOCK = 256

# Scoped optional distributed multiplier; unset on the ordinary single-GPU path.
_distributed_multiplier = ContextVar("tabai_distributed_multiplier", default=None)

# Mul dispatch.  For operands whose per-column work la*lb is at most this many
# limb-pairs, an all-GPU schoolbook kernel (base-2^16 column sums fed into the
# shared carry-resolution pipeline) beats the transform: it has no plan/malloc
# overhead and no GPU->CPU->GPU roundtrip.  Above the threshold the O(n^2) column
# work loses to the NTT's O(n log n) and we switch to _mul_ntt.
#
# Tuned by sweep on an RTX 3090: for balanced (square) operands — the worst case
# for schoolbook at a fixed la*lb, since they maximise the longest column loop —
# the schoolbook/NTT crossover is L ~= 6016 limbs.  Below it schoolbook wins
# (~1.75x at L=3072); asymmetric operands of equal work win by more, so la*lb is
# a conservative routing metric.  isqrt(threshold) == 6016 exactly.  (It rose
# from 5120 when the float transform was replaced by the NTT: the NTT's
# forward-transform floor is a touch higher, so schoolbook stays ahead further.)
_MUL_SCHOOLBOOK_MAX_WORK = 6016 * 6016  # la*lb limb-pairs (36_192_256)

# divmod's reciprocal step. Below this size, computing floor(2^p / b) with
# Python's int // is faster than running ~log2(p/53) GPU NTT mul iterations.
# Above it, the CPU step dominates the whole divmod and the GPU Newton path wins.
# The threshold sits at the empirical crossover (~164 Kbit) for a/b with b≈a/2.
# It dropped from 12288 after the faster mul (schoolbook + squaring) and the
# leaner Newton ramp made the GPU path win from ~5120 limbs up (a swept crossover
# where cpu// and Newton tie; Newton wins outright by 5632).
_DIV_NEWTON_THRESHOLD_LIMBS = 5120  # ~164 Kbits

# In the Newton ramp, each step's approximation carries a couple of leading zero
# limbs on top.  Below this working width we drop them with a *deterministic*
# top-slice (no host sync): a stray zero limb makes the next small mul only
# marginally wider, and skipping ~log2(p/64) find_last_nonzero D2H syncs is the
# larger saving.  At or above it we _trim instead — there the extra limb can
# push the (now large) next mul past a transform-length power-of-two boundary
# and double its cost, which dwarfs the single sync the trim spends.  Swept.
_NEWTON_RAMP_TRIM_LIMBS = 40000

_compute_states_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_states(
    const unsigned int* a, int len_a,
    const unsigned int* b, int len_b,
    unsigned int* result, int* states,
    int n, int is_sub)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned long long av = (idx < len_a) ? (unsigned long long)a[idx] : 0ULL;
    unsigned long long bv = (idx < len_b) ? (unsigned long long)b[idx] : 0ULL;

    if (is_sub == 0) {
        unsigned long long s = av + bv;
        result[idx] = (unsigned int)s;
        states[idx] = (s > 0xFFFFFFFFULL) ? 2 : ((s == 0xFFFFFFFFULL) ? 1 : 0);
    } else {
        long long d = (long long)av - (long long)bv;
        result[idx] = (unsigned int)d;
        states[idx] = (d < 0) ? 2 : ((d == 0) ? 1 : 0);
    }
}
''', 'compute_states')

_block_scan_kernel = cp.RawKernel(r'''
extern "C" __global__
void block_scan(int* states, int* block_out, int n)
{
    extern __shared__ int sm[];
    int* buf0 = sm;
    int* buf1 = sm + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    buf0[tid] = (idx < n) ? states[idx] : 0;
    __syncthreads();

    int* src = buf0;
    int* dst = buf1;

    for (int step = 1; step < (int)blockDim.x; step <<= 1) {
        if (tid >= step && src[tid] == 1) {
            dst[tid] = src[tid - step];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }

    if (idx < n) states[idx] = src[tid];

    if (block_out && tid == (int)blockDim.x - 1) {
        int last_local = min((int)blockDim.x - 1, n - 1 - blockIdx.x * (int)blockDim.x);
        block_out[blockIdx.x] = src[last_local];
    }
}
''', 'block_scan')

_propagate_kernel = cp.RawKernel(r'''
extern "C" __global__
void propagate(int* states, const int* block_sums, int n, int bs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int blk = idx / bs;
    if (blk == 0) return;
    if (states[idx] == 1) {
        states[idx] = block_sums[blk - 1];
    }
}
''', 'propagate')

_apply_carries_kernel = cp.RawKernel(r'''
extern "C" __global__
void apply_carries(unsigned int* result, const int* states, int n, int is_sub)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx > n) return;
    if (idx < n) {
        if (states[idx - 1] == 2) {
            if (is_sub == 0) result[idx] += 1u;
            else result[idx] -= 1u;
        }
    } else {
        // idx == n: always write carry-out for add so result can be cp.empty
        if (is_sub == 0) {
            result[n] = (states[n - 1] == 2) ? 1u : 0u;
        }
    }
}
''', 'apply_carries')

_find_last_nonzero_kernel = cp.RawKernel(r'''
extern "C" __global__
void find_last_nonzero(const unsigned int* arr, int n, int* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && arr[idx] != 0) {
        atomicMax(out, idx);
    }
}
''', 'find_last_nonzero')

_compare_kernel = cp.RawKernel(r'''
extern "C" __global__
void compare_arrays(
    const unsigned int* a, int len_a,
    const unsigned int* b, int len_b,
    unsigned long long* result, int n)
{
    __shared__ unsigned long long block_max;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (tid == 0) block_max = 0ULL;
    __syncthreads();

    if (idx < n) {
        unsigned int av = (idx < len_a) ? a[idx] : 0u;
        unsigned int bv = (idx < len_b) ? b[idx] : 0u;
        if (av != bv) {
            unsigned long long enc = ((unsigned long long)(unsigned int)(idx + 1) << 1)
                                     | ((av > bv) ? 1ULL : 0ULL);
            atomicMax(&block_max, enc);
        }
    }
    __syncthreads();

    if (tid == 0 && block_max != 0ULL) {
        atomicMax(result, block_max);
    }
}
''', 'compare_arrays')

_shift_right_one_kernel = cp.RawKernel(r'''
extern "C" __global__
void shift_right_one(const unsigned int* src, unsigned int* dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int val = src[idx] >> 1;
    if (idx + 1 < n) {
        val |= (src[idx + 1] & 1u) << 31;
    }
    dst[idx] = val;
}
''', 'shift_right_one')

_shift_left_kernel = cp.RawKernel(r'''
extern "C" __global__
void shift_left(const unsigned int* src, int src_len,
                unsigned int* dst, int dst_len,
                int limb_shift, int bit_shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= src_len) return;
    unsigned long long val = (unsigned long long)src[idx] << bit_shift;
    int lo = idx + limb_shift;
    int hi = lo + 1;
    if (lo < dst_len)
        atomicAdd((unsigned int*)&dst[lo], (unsigned int)(val & 0xFFFFFFFF));
    if (bit_shift > 0 && hi < dst_len)
        atomicAdd((unsigned int*)&dst[hi], (unsigned int)(val >> 32));
}
''', 'shift_left')

# Fused add/sub kernel for n <= _BLOCK (single block fits in shared memory).
# Replaces compute_states + block_scan + apply_carries + find_last_nonzero (4→1 launch).
# Also writes the trim index to trim_out[0] so _trim's separate kernel is not needed.
#
# Shared memory layout (dynamic, all 4-byte elements):
#   [0 .. BLOCK-1]         int32  buf0  (state ping buffer)
#   [BLOCK .. 2*BLOCK-1]   int32  buf1  (state pong buffer)
#   [2*BLOCK .. 3*BLOCK-1] uint32 res   (raw limb results, n ≤ BLOCK slots)
# Plus one static __shared__ int last_nz for the inline trim reduction.
# ---------------------------------------------------------------------------
# Carry propagation kernel for transform-based multiplication.
#
# One "global parallel" iteration of carry propagation over a chunk array:
#   dst[i] = (src[i] & chunk_mask) + (src[i-1] >> chunk_bits)
#
# src and dst must be different arrays (ping-pong).  All reads from src
# happen before any writes to dst (separate arrays → no data hazard).
# Equivalent to one iteration of the original Python carry loop but
# without any GPU→CPU synchronisation.
# ---------------------------------------------------------------------------
_carry_prop_step_kernel = cp.RawKernel(r'''
extern "C" __global__
void carry_prop_step(
    const long long* __restrict__ src,
    long long* __restrict__ dst,
    int n, int chunk_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long long chunk_mask = (1LL << chunk_bits) - 1LL;
    long long carry_in = (idx > 0) ? (src[idx - 1] >> chunk_bits) : 0LL;
    dst[idx] = (src[idx] & chunk_mask) + carry_in;
}
''', 'carry_prop_step')

# ---------------------------------------------------------------------------
# After fixed-count carry propagation, each element is at most 2^chunk_bits.
# Extract the 3-state encoding used by the add/sub prefix scan:
#   2 = generate: this position already carries (value >> chunk_bits == 1)
#   1 = propagate: low bits are all-ones; adding carry-in 1 would overflow
#   0 = kill: no carry out regardless of carry-in
# ---------------------------------------------------------------------------
_extract_mul_states_kernel = cp.RawKernel(r'''
extern "C" __global__
void extract_mul_states(
    const long long* __restrict__ src,
    int* __restrict__ states,
    int n, int chunk_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long long chunk_mask = (1LL << chunk_bits) - 1LL;
    long long val = src[idx];
    int carry = (val >> chunk_bits) ? 1 : 0;
    long long chunk = val & chunk_mask;
    if (carry)
        states[idx] = 2;
    else if (chunk == chunk_mask)
        states[idx] = 1;
    else
        states[idx] = 0;
}
''', 'extract_mul_states')

# ---------------------------------------------------------------------------
# Apply prefix-scan-resolved carry states back to the int64 chunk array.
# After this kernel every element fits in exactly chunk_bits bits.
# ---------------------------------------------------------------------------
_apply_mul_carries_kernel = cp.RawKernel(r'''
extern "C" __global__
void apply_mul_carries(
    long long* __restrict__ data,
    const int* __restrict__ states,
    int n, int chunk_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long long chunk_mask = (1LL << chunk_bits) - 1LL;
    long long val = data[idx] & chunk_mask;
    if (idx > 0 && states[idx - 1] == 2) {
        val = (val + 1LL) & chunk_mask;
    }
    data[idx] = val;
}
''', 'apply_mul_carries')

# ---------------------------------------------------------------------------
# Phase 2: all-GPU schoolbook multiply producing base-2^16 column sums.
#
# One thread = one output column c.  Each uint32 limb is split into two 16-bit
# half-words (little-endian), and column c accumulates  sum_{i+j=c} a16[i]*b16[j].
# Every partial product aw*bw < 2^32; a column has at most min(2*na, 2*nb) terms,
# so the int64 accumulator is safe (min_chunks << 2^31 for any operand we route
# here — capped by _MUL_SCHOOLBOOK_MAX_WORK).  The int64 column sums feed the
# same carry-resolution pipeline (carry_prop_step / extract / scan / apply) as
# the NTT path, so no GPU->CPU roundtrip ever occurs.
# ---------------------------------------------------------------------------
_schoolbook_mul16_kernel = cp.RawKernel(r'''
extern "C" __global__
void schoolbook_mul16(
    const unsigned int* __restrict__ a, int na,
    const unsigned int* __restrict__ b, int nb,
    long long* __restrict__ out, int n_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    int na16 = 2 * na;
    int nb16 = 2 * nb;
    int i_lo = col - (nb16 - 1);
    if (i_lo < 0) i_lo = 0;
    int i_hi = col;
    if (i_hi > na16 - 1) i_hi = na16 - 1;
    long long acc = 0;
    for (int i = i_lo; i <= i_hi; ++i) {
        int j = col - i;
        unsigned int aw = (a[i >> 1] >> ((i & 1) * 16)) & 0xFFFFu;
        unsigned int bw = (b[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu;
        acc += (long long)(aw * bw);
    }
    out[col] = acc;
}
''', 'schoolbook_mul16')

_addsub_small_kernel = cp.RawKernel(r'''
extern "C" __global__
void addsub_small(
    const unsigned int* a, int len_a,
    const unsigned int* b, int len_b,
    unsigned int* result,
    int n, int extra, int is_sub,
    int* trim_out)
{
    extern __shared__ int sm[];
    int* buf0 = sm;
    int* buf1 = sm + blockDim.x;
    unsigned int* res = (unsigned int*)(sm + 2 * (int)blockDim.x);

    __shared__ int last_nz;

    int tid = threadIdx.x;

    // Step 1: compute per-limb state and raw result into shared memory
    if (tid < n) {
        unsigned long long av = (tid < len_a) ? (unsigned long long)a[tid] : 0ULL;
        unsigned long long bv = (tid < len_b) ? (unsigned long long)b[tid] : 0ULL;
        if (is_sub == 0) {
            unsigned long long s = av + bv;
            res[tid] = (unsigned int)s;
            buf0[tid] = (s > 0xFFFFFFFFULL) ? 2 : ((s == 0xFFFFFFFFULL) ? 1 : 0);
        } else {
            long long d = (long long)av - (long long)bv;
            res[tid] = (unsigned int)d;
            buf0[tid] = (d < 0) ? 2 : ((d == 0) ? 1 : 0);
        }
    } else {
        buf0[tid] = 0;
    }
    __syncthreads();

    // Step 2: intra-block prefix scan (double-buffer)
    int* src = buf0;
    int* dst = buf1;
    for (int step = 1; step < (int)blockDim.x; step <<= 1) {
        if (tid >= step && src[tid] == 1) {
            dst[tid] = src[tid - step];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }
    // src now holds the fully propagated states

    // Step 3: apply carry/borrow
    if (tid > 0 && tid < n && src[tid - 1] == 2) {
        if (is_sub == 0) res[tid] += 1u;
        else             res[tid] -= 1u;
    }
    __syncthreads();

    // Step 4: write result to global memory + carry-out + inline trim
    if (tid == 0) {
        if (extra > 0 && is_sub == 0 && n > 0 && src[n - 1] == 2) {
            result[n] = 1u;
            last_nz = n;   // carry-out is the highest non-zero slot
        } else {
            if (extra > 0) result[n] = 0u;
            last_nz = -1;
        }
    }
    __syncthreads();

    if (tid < n) {
        result[tid] = res[tid];
        if (res[tid] != 0) atomicMax(&last_nz, tid);
    }
    __syncthreads();

    if (tid == 0) trim_out[0] = last_nz;
}
''', 'addsub_small')


# ===========================================================================
# NTT-based multiplication over the Goldilocks prime p = 2^64 - 2^32 + 1.
#
# Every coefficient is computed mod p, so there is no rounding and the product
# is exact at any size — unlike a float convolution, which has to shrink the
# chunk width at scale to keep the inverse-transform coefficients inside
# float64's exact-integer range.
#
# p has 2-adicity 32 (p-1 = 2^32 * (2^32-1)), so any power-of-two transform
# length up to 2^32 has a primitive root; 7 is a primitive root of p.  128->64
# bit reduction uses 2^64 = 2^32-1 and 2^96 = -1 (mod p) — a couple of branches
# and one multiply, no Montgomery/Barrett.  Column sums land in one uint64 and
# feed the same carry-resolution pipeline as the schoolbook path.
# ---------------------------------------------------------------------------
_GL_P = (1 << 64) - (1 << 32) + 1        # 0xFFFFFFFF00000001, prime
_GL_PRIMITIVE_ROOT = 7

# Shared device functions prepended to every NTT kernel source.
_GL_PREAMBLE = r'''
#define GL_P   18446744069414584321ULL   /* 2^64 - 2^32 + 1 */
#define GL_EPS 0xFFFFFFFFULL             /* 2^32 - 1  ( = 2^64 mod p) */

/* Reduce a 128-bit value hi*2^64 + lo into [0, p).  Valid for ANY 128-bit
   input (the hi/lo halves need not be canonical).  Uses
       2^64 = 2^32 - 1  and  2^96 = -1  (mod p),
   so with hi = h1*2^32 + h0:  x = lo + (2^32-1)*h0 - h1  (mod p). */
__device__ __forceinline__ unsigned long long gl_reduce128(
        unsigned long long hi, unsigned long long lo)
{
    unsigned long long h0 = hi & GL_EPS;
    unsigned long long h1 = hi >> 32;
    unsigned long long t  = lo - h1;
    if (lo < h1) t -= GL_EPS;            /* borrow: wrapped by +2^64 = +EPS */
    unsigned long long u = h0 * GL_EPS;  /* < 2^64, exact */
    unsigned long long r = t + u;
    if (r < t) r += GL_EPS;              /* carry: wrapped by -2^64 = -EPS */
    if (r >= GL_P) r -= GL_P;
    return r;
}

__device__ __forceinline__ unsigned long long gl_mulmod(
        unsigned long long a, unsigned long long b)
{
    return gl_reduce128(__umul64hi(a, b), a * b);
}

/* add/sub assume canonical inputs (< p) and return canonical outputs. */
__device__ __forceinline__ unsigned long long gl_addmod(
        unsigned long long a, unsigned long long b)
{
    unsigned long long s = a + b;
    if (s < a) s += GL_EPS;              /* overflow past 2^64 = +EPS mod p */
    if (s >= GL_P) s -= GL_P;
    return s;
}

__device__ __forceinline__ unsigned long long gl_submod(
        unsigned long long a, unsigned long long b)
{
    unsigned long long s = a - b;
    if (a < b) s -= GL_EPS;              /* borrow: -2^64 = -EPS mod p */
    return s;
}
'''

# Fill w[k] = base^k mod p for k in [0, m).  One thread per k, square-and-
# multiply over the (<= 31-bit) exponent k.  Used to build twiddle tables.
_gl_fill_powers_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void gl_fill_powers(unsigned long long* w, unsigned long long base, int m)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    unsigned long long result = 1ULL;
    unsigned long long b = base;
    unsigned int e = (unsigned int)k;
    while (e) {
        if (e & 1u) result = gl_mulmod(result, b);
        b = gl_mulmod(b, b);
        e >>= 1;
    }
    w[k] = result;
}
''', 'gl_fill_powers')

# One DIF (Gentleman-Sande) butterfly stage, in-place.  idx in [0, n_half):
#   block = idx >> log_half,  j = idx & (half-1)
#   i0 = block*(2*half) + j,  i1 = i0 + half
# Twiddle index j*tw_stride < n_half = len(w).  Natural-order in -> bit-reversed
# out over the whole forward transform (stages run half = n/2 down to 1).
_ntt_dif_stage_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void ntt_dif_stage(unsigned long long* a, const unsigned long long* w,
                   int n_half, int half, int log_half, int tw_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half) return;
    int j = idx & (half - 1);
    size_t i0 = ((size_t)(idx >> log_half) << (log_half + 1)) | (size_t)j;
    size_t i1 = i0 + (size_t)half;
    unsigned long long u = a[i0];
    unsigned long long v = a[i1];
    a[i0] = gl_addmod(u, v);
    a[i1] = gl_mulmod(gl_submod(u, v), w[(size_t)j * (size_t)tw_stride]);
}
''', 'ntt_dif_stage')

# One DIT (Cooley-Tukey) inverse butterfly stage, in-place, with winv[k]=w^-k.
# Same index arithmetic as the DIF stage; stages run half = 1 up to n/2, taking
# bit-reversed input back to natural order.  The 1/n scaling is NOT applied
# here — it is folded into the pointwise kernel.
_ntt_dit_inv_stage_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void ntt_dit_inv_stage(unsigned long long* a, const unsigned long long* winv,
                       int n_half, int half, int log_half, int tw_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half) return;
    int j = idx & (half - 1);
    size_t i0 = ((size_t)(idx >> log_half) << (log_half + 1)) | (size_t)j;
    size_t i1 = i0 + (size_t)half;
    unsigned long long u = a[i0];
    unsigned long long t = gl_mulmod(a[i1], winv[(size_t)j * (size_t)tw_stride]);
    a[i0] = gl_addmod(u, t);
    a[i1] = gl_submod(u, t);
}
''', 'ntt_dit_inv_stage')

# Two global stages per pass. Each thread owns four coefficients throughout
# both butterflies, so the intermediate values stay in registers. This halves
# full-array traffic without requiring synchronization between thread blocks.
_ntt_dif_pair_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void ntt_dif_pair(unsigned long long* a, const unsigned long long* w,
                  int n_quarter, int quarter, int log_quarter, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_quarter) return;
    int j = idx & (quarter - 1);
    size_t i = ((size_t)(idx >> log_quarter) << (log_quarter + 2)) | (size_t)j;
    unsigned long long x0 = a[i], x1 = a[i + quarter];
    unsigned long long x2 = a[i + 2 * (size_t)quarter];
    unsigned long long x3 = a[i + 3 * (size_t)quarter];
    unsigned long long w0 = w[(size_t)j * stride];
    unsigned long long w1 = w[(size_t)(j + quarter) * stride];
    unsigned long long w2 = w[(size_t)j * (2 * (size_t)stride)];
    unsigned long long t0 = gl_addmod(x0, x2);
    unsigned long long t1 = gl_addmod(x1, x3);
    unsigned long long t2 = gl_mulmod(gl_submod(x0, x2), w0);
    unsigned long long t3 = gl_mulmod(gl_submod(x1, x3), w1);
    a[i] = gl_addmod(t0, t1);
    a[i + quarter] = gl_mulmod(gl_submod(t0, t1), w2);
    a[i + 2 * (size_t)quarter] = gl_addmod(t2, t3);
    a[i + 3 * (size_t)quarter] = gl_mulmod(gl_submod(t2, t3), w2);
}
''', 'ntt_dif_pair')

_ntt_dit_inv_pair_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void ntt_dit_inv_pair(unsigned long long* a, const unsigned long long* w,
                      int n_quarter, int quarter, int log_quarter, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_quarter) return;
    int j = idx & (quarter - 1);
    size_t i = ((size_t)(idx >> log_quarter) << (log_quarter + 2)) | (size_t)j;
    unsigned long long x0 = a[i], x1 = a[i + quarter];
    unsigned long long x2 = a[i + 2 * (size_t)quarter];
    unsigned long long x3 = a[i + 3 * (size_t)quarter];
    unsigned long long w0 = w[(size_t)j * stride];
    unsigned long long w1 = w[(size_t)(j + quarter) * stride];
    unsigned long long w2 = w[(size_t)j * (2 * (size_t)stride)];
    x1 = gl_mulmod(x1, w2);
    x3 = gl_mulmod(x3, w2);
    unsigned long long t0 = gl_addmod(x0, x1);
    unsigned long long t1 = gl_submod(x0, x1);
    unsigned long long t2 = gl_mulmod(gl_addmod(x2, x3), w0);
    unsigned long long t3 = gl_mulmod(gl_submod(x2, x3), w1);
    a[i] = gl_addmod(t0, t2);
    a[i + quarter] = gl_addmod(t1, t3);
    a[i + 2 * (size_t)quarter] = gl_submod(t0, t2);
    a[i + 3 * (size_t)quarter] = gl_submod(t1, t3);
}
''', 'ntt_dit_inv_pair')

# The final forward / initial inverse stages only communicate within a tile.
# Read/write global coefficients once and use block barriers in shared memory
# between these stages. The twiddle stride still refers to the full transform.
_NTT_TILE = 1024
_ntt_tile_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void ntt_tile(unsigned long long* a, const unsigned long long* w,
              int n, int tile, int inverse)
{
    extern __shared__ unsigned long long values[];
    int t = threadIdx.x;
    size_t base = (size_t)blockIdx.x * tile;
    values[t] = a[base + t];
    values[t + tile / 2] = a[base + t + tile / 2];
    __syncthreads();
    if (inverse) {
        for (int half = 1; half < tile; half <<= 1) {
            int j = t & (half - 1);
            int i = 2 * (t - j) + j;
            unsigned long long u = values[i];
            unsigned long long v = gl_mulmod(values[i + half], w[(size_t)j * (n / (2 * half))]);
            values[i] = gl_addmod(u, v);
            values[i + half] = gl_submod(u, v);
            __syncthreads();
        }
    } else {
        for (int half = tile / 2; half; half >>= 1) {
            int j = t & (half - 1);
            int i = 2 * (t - j) + j;
            unsigned long long u = values[i], v = values[i + half];
            values[i] = gl_addmod(u, v);
            values[i + half] = gl_mulmod(gl_submod(u, v), w[(size_t)j * (n / (2 * half))]);
            __syncthreads();
        }
    }
    a[base + t] = values[t];
    a[base + t + tile / 2] = values[t + tile / 2];
}
''', 'ntt_tile')

# Fused pointwise product + 1/n scaling: a[k] <- a[k]*b[k]*scale mod p.
# b may alias a (squaring): the kernel only reads b, so it is safe.
_ntt_pointwise_scale_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void ntt_pointwise_scale(unsigned long long* a, const unsigned long long* b,
                         unsigned long long scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = gl_mulmod(gl_mulmod(a[idx], b[idx]), scale);
}
''', 'ntt_pointwise_scale')


class GPUBigInt:
    def __init__(self, max_bits=1_000_000):
        self.device_id = cp.cuda.runtime.getDevice()
        self._one = cp.array([1], dtype=cp.uint32)
        self._scan_dummy = cp.empty(1, dtype=cp.int32)
        self._trim_idx_buf = cp.empty(1, dtype=cp.int32)
        self._compare_buf = cp.empty(1, dtype=cp.uint64)
        self._states_buf = cp.empty(0, dtype=cp.int32)  # grown lazily
        self._scan_bufs: list[cp.ndarray] = []
        # int64 carry ping-pong buffers, shared by the schoolbook and NTT mul
        # paths.  Grown lazily by _ensure_carry_capacity(); never shrunk.
        self._carry_ping = cp.empty(0, dtype=cp.int64)
        self._carry_pong = cp.empty(0, dtype=cp.int64)
        # NTT workspace (uint64 mod-p coefficient pads) and twiddle-table cache
        # keyed by transform length n.  Grown lazily by _ensure_ntt_capacity().
        self._ntt_buf_a = cp.empty(0, dtype=cp.uint64)
        self._ntt_buf_b = cp.empty(0, dtype=cp.uint64)
        self._ntt_tables: dict[int, tuple] = {}
        self._ensure_capacity((max_bits + 31) // 32)

    def _ensure_capacity(self, n: int) -> None:
        """Grow pre-allocated working buffers so they can handle n limbs.

        Called at the start of any operation that uses the large-path buffers.
        Reallocation is amortised: we grow to the new size and never shrink.
        """
        if n <= len(self._states_buf):
            return
        self._states_buf = cp.empty(n, dtype=cp.int32)
        self._scan_bufs = []
        size = n
        while True:
            blocks = (size + _BLOCK - 1) // _BLOCK
            if blocks <= 1:
                break
            self._scan_bufs.append(cp.empty(blocks, dtype=cp.int32))
            size = blocks

    def _ensure_carry_capacity(self, carry_n: int) -> None:
        """Grow the int64 carry ping-pong buffers to hold carry_n chunk sums.

        Shared by the schoolbook and NTT mul paths (both write base-2^16 column
        sums here for the carry-resolution pipeline).  Amortised: grows but
        never shrinks.
        """
        if carry_n <= len(self._carry_ping):
            return
        self._carry_ping = cp.empty(carry_n, dtype=cp.int64)
        self._carry_pong = cp.empty(carry_n, dtype=cp.int64)

    def _ensure_ntt_capacity(self, n: int) -> None:
        """Grow uint64 transform pads. Carry storage uses the shorter linear
        convolution length, not the power-of-two padded transform length."""
        if n > len(self._ntt_buf_a):
            self._ntt_buf_a = cp.empty(n, dtype=cp.uint64)
            self._ntt_buf_b = cp.empty(n, dtype=cp.uint64)

    def _get_ntt_tables(self, n: int):
        """Return (w_fwd, w_inv, inv_n) for transform length n (a power of two).

        w_fwd[k] = w_n^k and w_inv[k] = w_n^-k for k in [0, n/2), where w_n is a
        primitive n-th root of unity mod p; inv_n = n^-1 mod p (Python int).
        Cached per n like a transform plan — the roots are generated once on the
        GPU by the square-and-multiply fill kernel."""
        cached = self._ntt_tables.get(n)
        if cached is not None:
            return cached
        half = max(1, n // 2)
        w_n = pow(_GL_PRIMITIVE_ROOT, (_GL_P - 1) // n, _GL_P)
        w_n_inv = pow(w_n, _GL_P - 2, _GL_P)
        inv_n = pow(n, _GL_P - 2, _GL_P)
        w_fwd = cp.empty(half, dtype=cp.uint64)
        w_inv = cp.empty(half, dtype=cp.uint64)
        blocks = (half + _BLOCK - 1) // _BLOCK
        _gl_fill_powers_kernel((blocks,), (_BLOCK,), (w_fwd, np.uint64(w_n), half))
        _gl_fill_powers_kernel((blocks,), (_BLOCK,), (w_inv, np.uint64(w_n_inv), half))
        tables = (w_fwd, w_inv, inv_n)
        self._ntt_tables[n] = tables
        return tables

    def _ntt_forward(self, a, n, w_fwd):
        """In-place DIF: register-fused global stages, then a shared tile."""
        if n < 2:
            return
        tile = min(n, _NTT_TILE)
        half = n // 2
        log_half = n.bit_length() - 2
        while half >= tile:
            if half >= 2 * tile:
                _ntt_dif_pair_kernel(
                    (((n // 4) + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                    (a, w_fwd, n // 4, half // 2, log_half - 1, n // (2 * half)))
                half >>= 2
                log_half -= 2
            else:
                _ntt_dif_stage_kernel(
                    (((n // 2) + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                    (a, w_fwd, n // 2, half, log_half, n // (2 * half)))
                half >>= 1
                log_half -= 1
        _ntt_tile_kernel((n // tile,), (tile // 2,),
                         (a, w_fwd, n, tile, 0), shared_mem=tile * 8)

    def _ntt_inverse(self, a, n, w_inv):
        """In-place DIT: shared tile then global pairs; scaling is separate."""
        if n < 2:
            return
        tile = min(n, _NTT_TILE)
        _ntt_tile_kernel((n // tile,), (tile // 2,),
                         (a, w_inv, n, tile, 1), shared_mem=tile * 8)
        half = tile
        log_half = tile.bit_length() - 1
        while half < n:
            if 4 * half <= n:
                _ntt_dit_inv_pair_kernel(
                    (((n // 4) + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                    (a, w_inv, n // 4, half, log_half, n // (4 * half)))
                half <<= 2
                log_half += 2
            else:
                _ntt_dit_inv_stage_kernel(
                    (((n // 2) + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                    (a, w_inv, n // 2, half, log_half, n // (2 * half)))
                half <<= 1
                log_half += 1

    def _mul_ntt(self, a_gpu, b_gpu, is_square=False):
        """Exact big-integer multiply via NTT over the Goldilocks prime.

        uint32 little-endian limb arrays in, a uint32 limb array (<= la+lb
        limbs) out.  16-bit chunks are convolved mod p (no rounding, exact at
        any size), then the resulting base-2^16 column sums run through the
        shared carry-resolution pipeline.  is_square (a_gpu is b_gpu) does one
        forward transform and squares it pointwise."""
        chunk_bits = 16
        a_ch = cp.ascontiguousarray(a_gpu).view(cp.uint16)
        b_ch = a_ch if is_square else cp.ascontiguousarray(b_gpu).view(cp.uint16)
        n_a, n_b = len(a_ch), len(b_ch)
        # Column sums are < min_chunks*(2^16-1)^2; this keeps them < 2^63 for the
        # int64 carry pipeline (also << p, so the mod-p transform is exact).
        assert min(n_a, n_b) * (0xFFFF ** 2) < (1 << 63)

        n_conv = n_a + n_b - 1
        n = 1 << max(0, n_conv - 1).bit_length()   # next power of two >= n_conv

        w_fwd, w_inv, inv_n = self._get_ntt_tables(n)
        self._ensure_ntt_capacity(n)

        fa = self._ntt_buf_a[:n]
        fa[:n_a] = a_ch          # cast uint16 -> uint64 on assignment
        fa[n_a:] = 0             # clear pad tail (buffer is reused)
        self._ntt_forward(fa, n, w_fwd)
        if is_square:
            fb = fa
        else:
            fb = self._ntt_buf_b[:n]
            fb[:n_b] = b_ch
            fb[n_b:] = 0
            self._ntt_forward(fb, n, w_fwd)   # forward uses w_fwd for both

        blocks = (n + _BLOCK - 1) // _BLOCK
        _ntt_pointwise_scale_kernel(
            (blocks,), (_BLOCK,), (fa, fb, np.uint64(inv_n), n))
        self._ntt_inverse(fa, n, w_inv)

        # Feed the exact column sums into the shared carry pipeline.
        # Cyclic-safe padding guarantees coefficients beyond n_conv are zero.
        # Resolve only the linear convolution and its carry-out slot, avoiding
        # whole-array carry passes over the power-of-two padding.
        carry_n = n_conv + 1
        self._ensure_carry_capacity(carry_n)
        ping = self._carry_ping[:carry_n]
        ping[:n_conv] = fa[:n_conv]  # uint64 -> int64; values < 2^63
        ping[n_conv] = 0
        max_bits = 2 * chunk_bits + min(n_a, n_b).bit_length()
        ping = self._resolve_carries(carry_n, chunk_bits, max_bits)
        return self._recombine_chunks(ping, carry_n, chunk_bits,
                                      len(a_gpu) + len(b_gpu))

    def _scan(self, states, n, _level=0):
        blocks = (n + _BLOCK - 1) // _BLOCK
        smem = 2 * _BLOCK * 4

        if blocks == 1:
            _block_scan_kernel((1,), (_BLOCK,), (states, self._scan_dummy, n), shared_mem=smem)
            return

        if _level < len(self._scan_bufs):
            block_sums = self._scan_bufs[_level][:blocks]
        else:
            block_sums = cp.empty(blocks, dtype=cp.int32)  # fallback for very large inputs

        _block_scan_kernel((blocks,), (_BLOCK,), (states, block_sums, n), shared_mem=smem)
        self._scan(block_sums, blocks, _level + 1)
        _propagate_kernel((blocks,), (_BLOCK,), (states, block_sums, n, _BLOCK))

    def _addsub(self, a_gpu, b_gpu, is_sub):
        n = max(len(a_gpu), len(b_gpu))
        extra = 0 if is_sub else 1
        # cp.empty avoids cudaMemset; all slots are written explicitly by the kernels
        result = cp.empty(n + extra, dtype=cp.uint32)

        if n <= _BLOCK:
            # Fast path: single fused kernel (compute + carry-propagate).
            smem = 3 * _BLOCK * 4  # 2 state buffers + 1 result buffer
            _addsub_small_kernel(
                (1,), (_BLOCK,),
                (a_gpu, len(a_gpu), b_gpu, len(b_gpu), result, n, extra, int(is_sub),
                 self._trim_idx_buf),
                shared_mem=smem)
        else:
            self._ensure_capacity(n)
            states = self._states_buf[:n]
            blocks = (n + _BLOCK - 1) // _BLOCK
            _compute_states_kernel(
                (blocks,), (_BLOCK,),
                (a_gpu, len(a_gpu), b_gpu, len(b_gpu), result, states, n, int(is_sub)))
            self._scan(states, n)
            blocks_ext = ((n + extra) + _BLOCK - 1) // _BLOCK
            _apply_carries_kernel(
                (blocks_ext,), (_BLOCK,),
                (result, states, n, int(is_sub)))
        return result

    def add(self, a_gpu, b_gpu):
        return self._addsub(a_gpu, b_gpu, False)

    def sub(self, a_gpu, b_gpu):
        return self._addsub(a_gpu, b_gpu, True)

    def _addsub_trimmed(self, a_gpu, b_gpu, is_sub):
        """add/sub returning (trimmed_result, is_zero) for the TabaiInt layer.

        On the fused single-block path the trim index is already sitting in
        _trim_idx_buf (addsub_small writes it), so we consume it directly and
        skip a second find_last_nonzero launch plus the constructor's re-trim.
        The raw add/sub above stay sync-free and are used internally (Newton).
        """
        result = self._addsub(a_gpu, b_gpu, is_sub)
        n = max(len(a_gpu), len(b_gpu))
        if n <= _BLOCK:
            last = int(self._trim_idx_buf[0])  # trim index from addsub_small
            if last < 0:
                return cp.array([0], dtype=cp.uint32), True
            return result[:last + 1], False
        return self._trim_z(result)

    def add_trimmed(self, a_gpu, b_gpu):
        return self._addsub_trimmed(a_gpu, b_gpu, False)

    def sub_trimmed(self, a_gpu, b_gpu):
        return self._addsub_trimmed(a_gpu, b_gpu, True)

    def mul(self, a_gpu, b_gpu):
        # Dispatch.  For small per-column work an all-GPU schoolbook kernel
        # avoids the transform's plan/malloc overhead and any host roundtrip;
        # above the threshold the O(n log n) NTT wins over O(n^2) columns.
        la, lb = len(a_gpu), len(b_gpu)
        if la * lb <= _MUL_SCHOOLBOOK_MAX_WORK:
            return self._mul_schoolbook(a_gpu, b_gpu)
        distributed = _distributed_multiplier.get()
        if distributed is not None and 32 * max(la, lb) >= distributed.min_bits:
            return distributed.mul(self, a_gpu, b_gpu)
        # Squaring (same array object for both operands, as produced by __pow__'s
        # repeated squarings) lets the transform skip the second operand fill and
        # its forward transform — one forward NTT instead of two.
        return self._mul_ntt(a_gpu, b_gpu, is_square=a_gpu is b_gpu)

    def _resolve_carries(self, carry_n, chunk_bits, max_bits):
        """Resolve base-2^chunk_bits column sums into final chunk values.

        The caller must have already written the (arbitrary-magnitude) int64
        column sums into self._carry_ping[:carry_n], including a zeroed
        carry-out slot at the top.  Returns the buffer (ping or pong, depending
        on iteration parity) whose first carry_n int64 elements each hold a
        value < 2^chunk_bits.  Zero GPU->CPU syncs.

        max_bits is an upper bound on the bit length of the largest input column
        sum; it fixes the number of carry_prop_step iterations.  Over-estimating
        is harmless (once every element is < 2^chunk_bits the step is a no-op).
        """
        ping = self._carry_ping[:carry_n]
        pong = self._carry_pong[:carry_n]
        blocks_carry = (carry_n + _BLOCK - 1) // _BLOCK

        # Step 1: fixed-count carry reduction (no sync).
        k = -(-max_bits // chunk_bits)
        for _ in range(k):
            _carry_prop_step_kernel(
                (blocks_carry,), (_BLOCK,),
                (ping, pong, carry_n, chunk_bits))
            ping, pong = pong, ping

        # Step 2: extract 3-state encoding and parallel prefix-scan (no sync).
        self._ensure_capacity(carry_n)
        states = self._states_buf[:carry_n]
        _extract_mul_states_kernel(
            (blocks_carry,), (_BLOCK,),
            (ping, states, carry_n, chunk_bits))
        self._scan(states, carry_n)

        # Step 3: apply resolved carries (no sync).
        _apply_mul_carries_kernel(
            (blocks_carry,), (_BLOCK,),
            (ping, states, carry_n, chunk_bits))
        return ping

    def _recombine_chunks(self, ping, carry_n, chunk_bits, max_limbs):
        """Pack carry_n resolved base-2^chunk_bits chunks (int64, each
        < 2^chunk_bits) into uint32 limbs, conservatively trimmed to max_limbs.

        Writes into one padded buffer (no cp.concatenate whole-array copy):
        cast the chunks in, zeroing only the <= chunks_per_limb-1 pad tail.
        No GPU->CPU sync.
        """
        out_dtype = cp.uint16 if chunk_bits == 16 else cp.uint8
        chunks_per_limb = 32 // chunk_bits  # 2 for uint16, 4 for uint8
        n_padded = ((carry_n + chunks_per_limb - 1) // chunks_per_limb) * chunks_per_limb
        out = cp.empty(n_padded, dtype=out_dtype)
        if n_padded > carry_n:
            out[carry_n:] = 0
        out[:carry_n] = ping   # cast int64 -> uint16/uint8 on assignment
        limbs = out.view(cp.uint32)
        return limbs[:max_limbs] if len(limbs) > max_limbs else limbs

    def _mul_schoolbook(self, a_gpu, b_gpu):
        # base-2^16 column sums straight into the carry buffer, then the shared
        # carry-resolution pipeline.  No transform, no host roundtrip.
        chunk_bits = 16
        la, lb = len(a_gpu), len(b_gpu)
        na16, nb16 = 2 * la, 2 * lb
        n_cols = na16 + nb16 - 1
        carry_n = n_cols + 1
        self._ensure_carry_capacity(carry_n)
        ping = self._carry_ping[:carry_n]

        blocks = (n_cols + _BLOCK - 1) // _BLOCK
        _schoolbook_mul16_kernel(
            (blocks,), (_BLOCK,),
            (a_gpu, la, b_gpu, lb, ping, n_cols))
        ping[n_cols] = 0  # carry-out slot

        # Each column sum <= min(na16, nb16) * (2^16-1)^2 < 2^(2*16) * min(...);
        # its bit length is bounded by 2*chunk_bits + bit_length(min terms).
        n_terms = min(na16, nb16)
        max_bits = 2 * chunk_bits + n_terms.bit_length()
        ping = self._resolve_carries(carry_n, chunk_bits, max_bits)
        return self._recombine_chunks(ping, carry_n, chunk_bits, la + lb)

    def _trim_z(self, gpu_arr):
        """Trim leading zero limbs; also report whether the result is zero.

        The zero flag comes for free: it is exactly (last < 0) from the
        reduction the trim already performs.  Callers that need to know
        zero-ness (e.g. the TabaiInt layer's sign normalisation) thus avoid a
        second GPU→CPU sync.  Returns (trimmed_array, is_zero).
        """
        n = len(gpu_arr)
        if n == 0:
            return cp.array([0], dtype=cp.uint32), True
        # Reset pre-allocated buffer (async, serialized on the null stream)
        self._trim_idx_buf.fill(-1)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _find_last_nonzero_kernel((blocks,), (_BLOCK,), (gpu_arr, n, self._trim_idx_buf))
        last = int(self._trim_idx_buf[0])  # GPU→CPU sync
        if last < 0:
            return cp.array([0], dtype=cp.uint32), True
        return gpu_arr[:last + 1], False

    def _trim(self, gpu_arr):
        arr, _ = self._trim_z(gpu_arr)
        return arr

    def _compare(self, a_gpu, b_gpu):
        n = max(len(a_gpu), len(b_gpu))
        self._compare_buf.fill(0)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _compare_kernel(
            (blocks,), (_BLOCK,),
            (a_gpu, len(a_gpu), b_gpu, len(b_gpu), self._compare_buf, n))
        val = int(self._compare_buf[0])
        if val == 0:
            return 0
        return 1 if (val & 1) else -1

    def _bit_length(self, a_gpu):
        a = self._trim(a_gpu)
        if len(a) == 1 and int(a[0]) == 0:
            return 0
        return (len(a) - 1) * 32 + int(a[-1]).bit_length()

    def _shift_left(self, a_gpu, bits):
        if bits == 0:
            return a_gpu.copy()
        limb_shift = bits // 32
        bit_shift = bits % 32
        n = len(a_gpu) + limb_shift + (1 if bit_shift > 0 else 0)
        result = cp.zeros(n, dtype=cp.uint32)
        blocks = (len(a_gpu) + _BLOCK - 1) // _BLOCK
        _shift_left_kernel(
            (blocks,), (_BLOCK,),
            (a_gpu, len(a_gpu), result, n, limb_shift, bit_shift))
        return result

    def _shift_right_one(self, a_gpu):
        n = len(a_gpu)
        result = cp.empty(n, dtype=cp.uint32)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _shift_right_one_kernel((blocks,), (_BLOCK,), (a_gpu, result, n))
        return result

    def _to_int(self, gpu_arr):
        """Convert GPU uint32 little-endian limb array to Python int."""
        arr = cp.asnumpy(self._trim(gpu_arr))
        return int.from_bytes(arr.tobytes(), 'little')

    def _from_int(self, n):
        """Convert non-negative Python int to GPU uint32 little-endian limb array."""
        if n == 0:
            return cp.array([0], dtype=cp.uint32)
        byte_len = (n.bit_length() + 7) // 8
        b = n.to_bytes(byte_len, 'little')
        pad = (-len(b)) % 4
        if pad:
            b += b'\x00' * pad
        arr_np = np.frombuffer(b, dtype=np.uint32)
        return cp.asarray(arr_np)

    def _reciprocal(self, b, p):
        """Return floor(2^p / b) as a uint32 GPU array.

        Dispatches to the GPU Newton path for large p, otherwise computes the
        reciprocal on the CPU with Python big-int division.  Caller must
        guarantee b > 0, b already trimmed (top limb non-zero), and p % 32 == 0.
        """
        # b is trimmed and non-zero, so its bit length needs no re-trim — just
        # the top limb's bit_length (one unavoidable read).
        n_b = (len(b) - 1) * 32 + int(b[-1]).bit_length()
        if p < n_b:
            return cp.array([0], dtype=cp.uint32)

        # Newton needs at least 53 bits of headroom below p to seed from float64.
        # In other regimes the result has only a handful of bits anyway, so CPU //
        # is the right tool.
        if p < n_b + 53 or p // 32 <= _DIV_NEWTON_THRESHOLD_LIMBS:
            b_int = self._to_int(b)
            return self._from_int((1 << p) // b_int)
        return self._newton_reciprocal(b, p, n_b)

    def _newton_reciprocal(self, b, p, n_b):
        """Compute floor(2^p / b) via doubling-precision Newton iteration.

        A fixed-precision Newton needs ~log2(p/53) iterations, each a full
        scale-p multiply: ~log2(p) * M(p) work.  This version ramps the working
        precision so early iterations are cheap, costing only ~constant * M(p).

        The whole computation tracks x as an approximation of the real value
        2^e/b at a "b-exponent" e that is always a multiple of 32.  Because b is
        only ever truncated by dropping whole low limbs (b_trunc = floor(b/2^d),
        d a multiple of 32), every working scale stays 32-bit aligned, so each
        ">> scale" is a free little-endian array slice.  Framing x by its
        b-exponent makes the per-step change of truncation transparent: a
        differently-truncated b_trunc still approximates the same real 2^e/b.

        Steps:
          1. Exact CPU base seed (~64 quotient bits) from a small top slice of b.
             Being an exact floor division, the seed is fully accurate to its
             width (no leading-bit deficit), so accuracy doubles cleanly.
          2. Ramp: double the quotient precision each step via one Newton step,
             truncating b to just enough top limbs to cover the new precision.
          3. Finish: full-b, scale-p fixed-point iteration (the original proven
             iteration).  Newton maps x = r*(1+e) to r*(1-e^2) <= r*, so it
             self-corrects any overshoot the truncated ramp introduced and
             converges to the exact floor from below in ~1-2 iterations.
          4. Defensive +1 adjustment for an exact floor (same as before).

        Preconditions: b is trimmed, b > 0, p % 32 == 0, p >= n_b + 53.
        """
        p_limbs = p // 32

        # For tiny divisors the limb bookkeeping below has too little headroom;
        # defer to the CPU path (this regime is never the Newton threshold).
        if n_b < 64:
            b_int = self._to_int(b)
            return self._from_int((1 << p) // b_int)

        nb_limbs = len(b)          # b is trimmed, little-endian uint32
        target_q = p - n_b         # bit length of floor(2^p / b)
        GUARD = 3                  # extra top limbs of b kept beyond x precision

        def b_top(kb):
            """Top kb limbs of b as (int value, dropped low-bit count d)."""
            kb = min(nb_limbs, max(1, kb))
            d = (nb_limbs - kb) * 32
            return self._to_int(b[nb_limbs - kb:]), d

        # ---- Step 1: exact CPU base seed (~64 quotient bits) ----------------
        q0 = min(target_q, 64)
        ex = ((n_b + q0 + 31) // 32) * 32          # b-exponent, 32-aligned, <= p
        b_trunc_int, d = b_top((q0 + 31) // 32 + GUARD)
        x = self._from_int((1 << (ex - d)) // b_trunc_int)

        # ---- Step 2: doubling-precision ramp --------------------------------
        while ex < p:
            q_next = 2 * (ex - n_b)
            ex_next = ((n_b + q_next + 31) // 32) * 32
            if ex_next > p:
                ex_next = p

            # Lift x to the new b-exponent: x ~ 2^ex/b  ->  2^ex_next/b.
            # In little-endian, << is prepending low zero limbs.
            shift_limbs = (ex_next - ex) // 32
            if shift_limbs:
                x = cp.concatenate([cp.zeros(shift_limbs, dtype=cp.uint32), x])

            # Truncate b to the top kb limbs (a slice view; no host transfer —
            # the ramp Newton step is all on the GPU, unlike the base seed).
            q_field = ex_next - n_b
            kb = (q_field + 31) // 32 + GUARD
            if kb >= nb_limbs:
                b_trunc, d = b, 0
            else:
                b_trunc = b[nb_limbs - kb:]
                d = (nb_limbs - kb) * 32
            s = ex_next - d            # scale w.r.t. b_trunc, 32-aligned
            s_limbs = s // 32

            # One Newton step at scale s: x <- floor(x*(2^(s+1) - b_trunc*x)/2^s)
            bx = self.mul(b_trunc, x)
            two = cp.zeros(s_limbs + 1, dtype=cp.uint32)
            two[s_limbs] = 2
            diff = self.sub(two, bx)
            prod = self.mul(x, diff)
            # >> s is a free low-limb slice.  Then normalise x's width, size-gated:
            #   * small x: deterministic top slice, no host sync.  b_trunc keeps
            #     b's top bit, so x = floor(2^s / b_trunc) <= 2^(q_field+1); its
            #     bit length is at most q_field+2 (equality only for a power-of-two
            #     b_trunc), i.e. it always fits in q_field//32 + 2 limbs.  Keeping
            #     exactly that many drops the Newton step's extra leading zeros
            #     without a find_last_nonzero D2H read.
            #   * large x: _trim to the exact width.  Here a stray leading-zero
            #     limb could push the next mul past a transform-length power
            #     of two and double its cost, which outweighs the one sync.
            lo = prod[s_limbs:] if len(prod) > s_limbs \
                else cp.array([0], dtype=cp.uint32)
            if len(lo) <= _NEWTON_RAMP_TRIM_LIMBS:
                x = lo[:q_field // 32 + 2]
            else:
                x = self._trim(lo)

            ex = ex_next

        # ---- Step 3: single full-b Newton finish (exact, from below) --------
        # The ramp uses a low-limb-truncated b, so x may still be a hair off the
        # true 2^p/b.  A single full-precision Newton step is provably enough:
        #   * the ramp always reaches >= half the target precision (it doubles q
        #     each step until ex==p), so the relative error e satisfies
        #     e <= 2^-(q/2), and one step maps e -> e^2 < 2^-q, i.e. within 1 ULP;
        #   * the map x -> x*(2^(p+1)-b*x)/2^p converges monotonically from below
        #     (from x=r(1+-f) it lands at r(1-f^2) <= r), so x <= floor(2^p/b)
        #     afterwards, leaving only a small upward correction for step 4.
        # A fixed single iteration therefore removes both the old confirmation
        # iteration and its per-iteration compare (a D2H sync) — measured
        # finish_iters was 1-2 with the loop, so 1 useful step always sufficed.
        two_p_plus_1 = cp.zeros(p_limbs + 1, dtype=cp.uint32)
        two_p_plus_1[p_limbs] = 2
        bx = self.mul(b, x)
        diff = self.sub(two_p_plus_1, bx)
        prod = self.mul(x, diff)
        x = prod[p_limbs:] if len(prod) > p_limbs \
            else cp.array([0], dtype=cp.uint32)

        # ---- Step 4: +1 adjustment to the exact floor ----------------------
        # x <= floor(2^p/b) from the finish, within a couple of ULPs.  Increment
        # while (x+1)*b <= 2^p.  b*(x+1) = b*x + b, so after one initial mul we
        # extend by a cheap add per step instead of a full-precision multiply.
        two_p = cp.zeros(p_limbs + 1, dtype=cp.uint32)
        two_p[p_limbs] = 1
        bx = self.mul(b, x)
        for _ in range(4):
            bx_next = self.add(bx, b)
            if self._compare(bx_next, two_p) <= 0:
                x = self.add(x, self._one)
                bx = bx_next
            else:
                break
        return x

    def divmod(self, a_gpu, b_gpu):
        b = self._trim(b_gpu)
        a = self._trim(a_gpu)
        if len(b) == 1 and int(b[0]) == 0:
            raise ZeroDivisionError("division by zero")
        cmp = self._compare(a, b)
        if cmp < 0:
            # a is already trimmed and immutable, so it is the remainder as-is
            # (no defensive copy needed — no kernel ever writes to an operand).
            return cp.array([0], dtype=cp.uint32), a
        if cmp == 0:
            return cp.array([1], dtype=cp.uint32), cp.array([0], dtype=cp.uint32)

        # a is trimmed and a > b >= 1, so a is non-zero: its bit length needs no
        # re-trim, just the top limb's bit_length (one unavoidable read).
        a_bits = (len(a) - 1) * 32 + int(a[-1]).bit_length()

        # p = smallest multiple of 32 that is >= a_bits.
        # With p >= a_bits, the approximation q0 satisfies q-1 <= q0 <= q,
        # so at most one correction is needed after the main computation.
        p_limbs = (a_bits + 31) // 32
        p = p_limbs * 32

        # Compute reciprocal x = floor(2^p / b).  Below the threshold, Python's
        # big-int // is faster than launching ~25 GPU NTT muls; above it the
        # GPU Newton path wins by a wide margin.
        x = self._reciprocal(b, p)

        # q0 = floor(a * x / 2^p)
        # Since p = p_limbs * 32, the right-shift is a free array slice (little-endian).
        ax = self.mul(a, x)
        if len(ax) <= p_limbs:
            # Should not happen when a >= b, but handle defensively.
            q0 = cp.array([0], dtype=cp.uint32)
        else:
            q0 = self._trim(ax[p_limbs:])

        # r = a - q0 * b  (always >= 0 since q0 <= floor(a/b))
        q0b = self.mul(q0, b)
        r = self.sub(a, q0b)

        # q0 may be off by 1 (too low): if r >= b, correct once.
        if self._compare(r, b) >= 0:
            r = self.sub(r, b)
            q0 = self.add(q0, self._one)

        return self._trim(q0), self._trim(r)

    def floordiv(self, a_gpu, b_gpu):
        q, _ = self.divmod(a_gpu, b_gpu)
        return q

    def mod(self, a_gpu, b_gpu):
        _, r = self.divmod(a_gpu, b_gpu)
        return r
