import cupy as cp
import numpy as np

_BLOCK = 256

# For mul operands whose combined size is at most this many limbs, delegate to
# Python's built-in int * (Karatsuba) instead of launching cuFFT.  The cuFFT
# pipeline costs ~1 ms regardless of input size (kernel-launch + plan overhead),
# while Python int * for 64 K-bit operands is ~900 µs and for 1 K-bit is ~2 µs.
# The crossover (FFT faster than Python) is around 80 K bits per operand.
_MUL_CPU_THRESHOLD_LIMBS = 2048  # 65536 bits per operand

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int av = (idx < len_a) ? a[idx] : 0u;
    unsigned int bv = (idx < len_b) ? b[idx] : 0u;
    if (av != bv) {
        unsigned long long enc = ((unsigned long long)(unsigned int)(idx + 1) << 1)
                                 | ((av > bv) ? 1ULL : 0ULL);
        atomicMax(result, enc);
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
# Carry propagation kernel for FFT-based multiplication (Phase 2).
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



class GPUBigInt:
    def __init__(self, max_bits=1_000_000):
        self._scan_dummy = cp.empty(1, dtype=cp.int32)
        self._trim_idx_buf = cp.empty(1, dtype=cp.int32)
        self._states_buf = cp.empty(0, dtype=cp.int32)  # grown lazily
        self._scan_bufs: list[cp.ndarray] = []
        # Phase 4: pre-allocated FFT workspace — avoids cudaMalloc per mul call.
        # Grown lazily by _ensure_fft_capacity(); never shrunk.
        self._fft_buf_a = cp.empty(0, dtype=cp.float64)
        self._fft_buf_b = cp.empty(0, dtype=cp.float64)
        self._fft_carry_ping = cp.empty(0, dtype=cp.int64)
        self._fft_carry_pong = cp.empty(0, dtype=cp.int64)
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

    def _ensure_fft_capacity(self, n_fft: int) -> None:
        """Grow pre-allocated FFT workspace buffers to handle transforms of size n_fft.

        Allocates:
          - Two float64 buffers of length n_fft  (FFT input pads for a and b).
          - Two int64 carry ping-pong buffers of length n_fft + 1  (one carry-out
            slot beyond the n_fft convolution coefficients).
        Reallocation is amortised: buffers grow but never shrink.
        """
        if n_fft <= len(self._fft_buf_a):
            return
        self._fft_buf_a = cp.empty(n_fft, dtype=cp.float64)
        self._fft_buf_b = cp.empty(n_fft, dtype=cp.float64)
        self._fft_carry_ping = cp.empty(n_fft + 1, dtype=cp.int64)
        self._fft_carry_pong = cp.empty(n_fft + 1, dtype=cp.int64)

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

    def mul(self, a_gpu, b_gpu):
        # ------------------------------------------------------------------ #
        # Fast path: small operands — Python int * beats cuFFT for < 64 K bits
        # ------------------------------------------------------------------ #
        if len(a_gpu) <= _MUL_CPU_THRESHOLD_LIMBS and len(b_gpu) <= _MUL_CPU_THRESHOLD_LIMBS:
            a_np = cp.asnumpy(a_gpu).astype(np.uint32)
            b_np = cp.asnumpy(b_gpu).astype(np.uint32)
            a_int = int.from_bytes(a_np.tobytes(), 'little')
            b_int = int.from_bytes(b_np.tobytes(), 'little')
            return self._from_int(a_int * b_int)

        # ------------------------------------------------------------------ #
        # Adaptive chunk width to maintain float64 precision.
        #
        # The FFT convolution computes coefficients whose maximum magnitude is
        # bounded by  n_fft × (2^B - 1)^2 ≈ n_fft × 2^(2B).
        # float64 can represent integers up to 2^53 exactly, so safe rounding
        # of the IFFT output requires:
        #
        #   n_fft × 2^(2B) < 2^52   (1 bit headroom for FFT rounding errors)
        #
        # With B = 16:  safe when n_fft < 2^20  (~4M-bit operands).
        # With B =  8:  safe when n_fft < 2^36  (operands up to ~10 Gbits).
        #
        # Estimate n_fft using the worst case (B = 16) to decide which path
        # to take; the actual n_fft for B = 8 is 2× larger but still safe.
        # ------------------------------------------------------------------ #
        n_chunks_16_est = (len(a_gpu) + len(b_gpu)) * 2
        n_fft_est = 1
        while n_fft_est < n_chunks_16_est:
            n_fft_est <<= 1

        if n_fft_est < (1 << 20):
            chunk_bits = 16
            # uint32 viewed as uint16 gives [lo, hi] per limb (little-endian)
            a_chunks = cp.ascontiguousarray(a_gpu).view(cp.uint16)
            b_chunks = cp.ascontiguousarray(b_gpu).view(cp.uint16)
        else:
            chunk_bits = 8
            # uint32 viewed as uint8 gives [b0, b1, b2, b3] per limb
            a_chunks = cp.ascontiguousarray(a_gpu).view(cp.uint8)
            b_chunks = cp.ascontiguousarray(b_gpu).view(cp.uint8)

        n_a = len(a_chunks)
        n_b = len(b_chunks)
        n_conv = n_a + n_b - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft <<= 1

        # Phase 4: ensure pre-allocated workspace is large enough (no malloc per call).
        self._ensure_fft_capacity(n_fft)

        # Phase 4: reuse pre-allocated float64 pads — clear only the padding region.
        a_f = self._fft_buf_a[:n_fft]
        b_f = self._fft_buf_b[:n_fft]
        a_f[:n_a] = a_chunks.astype(cp.float64)
        a_f[n_a:] = 0
        b_f[:n_b] = b_chunks.astype(cp.float64)
        b_f[n_b:] = 0

        fa = cp.fft.rfft(a_f)
        fb = cp.fft.rfft(b_f)
        fa *= fb
        c = cp.fft.irfft(fa, n=n_fft)

        # ------------------------------------------------------------------ #
        # Carry propagation — zero GPU→CPU syncs.
        #
        # Step 1: Fixed-count carry_prop_step iterations reduce every
        #   element from up to ~(log2(n_fft) + 2B) bits down to at most
        #   2^chunk_bits.  At that point each position's carry is 0 or 1.
        #
        # Step 2: Extract 3-state encoding (kill/propagate/generate) and
        #   resolve carry chains with the same parallel prefix scan used
        #   for add/sub.  O(log n) work, no GPU→CPU sync.
        #
        # Step 3: Apply resolved carries to produce final chunk values.
        # ------------------------------------------------------------------ #
        carry_n = n_fft + 1
        ping = self._fft_carry_ping[:carry_n]
        pong = self._fft_carry_pong[:carry_n]
        ping[:n_fft] = cp.rint(c).astype(cp.int64)
        ping[n_fft] = 0  # carry-out slot

        # Step 1: fixed-count carry reduction (no sync).
        k = -(-((n_fft.bit_length() - 1) + 2 * chunk_bits) // chunk_bits)
        blocks_carry = (carry_n + _BLOCK - 1) // _BLOCK
        for _ in range(k):
            _carry_prop_step_kernel(
                (blocks_carry,), (_BLOCK,),
                (ping, pong, carry_n, chunk_bits))
            ping, pong = pong, ping

        # Step 2: extract states and prefix-scan (no sync).
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

        # Recombine: view small-int array as uint32 limbs.
        out_dtype = cp.uint16 if chunk_bits == 16 else cp.uint8
        chunks_per_limb = 32 // chunk_bits  # 2 for uint16, 4 for uint8
        out = ping.astype(out_dtype)
        pad = (-len(out)) % chunks_per_limb
        if pad:
            out = cp.concatenate([out, cp.zeros(pad, dtype=out_dtype)])
        # Conservative trim: product of two numbers needs at most
        # len(a) + len(b) limbs.  Slice without any GPU→CPU sync.
        limbs = out.view(cp.uint32)
        max_limbs = len(a_gpu) + len(b_gpu)
        return limbs[:max_limbs] if len(limbs) > max_limbs else limbs

    def _trim(self, gpu_arr):
        n = len(gpu_arr)
        if n == 0:
            return cp.array([0], dtype=cp.uint32)
        # Reset pre-allocated buffer (async, serialized on the null stream)
        self._trim_idx_buf.fill(-1)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _find_last_nonzero_kernel((blocks,), (_BLOCK,), (gpu_arr, n, self._trim_idx_buf))
        last = int(self._trim_idx_buf[0])  # GPU→CPU sync
        if last < 0:
            return cp.array([0], dtype=cp.uint32)
        return gpu_arr[:last + 1]

    def _compare(self, a_gpu, b_gpu):
        n = max(len(a_gpu), len(b_gpu))
        result_buf = cp.zeros(1, dtype=cp.uint64)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _compare_kernel(
            (blocks,), (_BLOCK,),
            (a_gpu, len(a_gpu), b_gpu, len(b_gpu), result_buf, n))
        val = int(result_buf[0])
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

    def divmod(self, a_gpu, b_gpu):
        b = self._trim(b_gpu)
        a = self._trim(a_gpu)
        if len(b) == 1 and int(b[0]) == 0:
            raise ZeroDivisionError("division by zero")
        cmp = self._compare(a, b)
        if cmp < 0:
            return cp.array([0], dtype=cp.uint32), a.copy()
        if cmp == 0:
            return cp.array([1], dtype=cp.uint32), cp.array([0], dtype=cp.uint32)

        a_bits = self._bit_length(a)

        # p = smallest multiple of 32 that is >= a_bits.
        # With p >= a_bits, the approximation q0 satisfies q-1 <= q0 <= q,
        # so at most one correction is needed after the main computation.
        p_limbs = (a_bits + 31) // 32
        p = p_limbs * 32

        # Compute reciprocal x = floor(2^p / b) using Python integer arithmetic.
        # Python's big-int // uses fast algorithms (Karatsuba etc.) internally.
        # The round-trip (GPU→CPU→GPU) is justified: b is transferred once, x once,
        # while all expensive multiplications (a*x, q0*b) stay on the GPU.
        b_cpu = self._to_int(b)
        x_cpu = (1 << p) // b_cpu
        x = self._from_int(x_cpu)

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
            q0 = self.add(q0, cp.array([1], dtype=cp.uint32))

        return self._trim(q0), self._trim(r)

    def floordiv(self, a_gpu, b_gpu):
        q, _ = self.divmod(a_gpu, b_gpu)
        return q

    def mod(self, a_gpu, b_gpu):
        _, r = self.divmod(a_gpu, b_gpu)
        return r
