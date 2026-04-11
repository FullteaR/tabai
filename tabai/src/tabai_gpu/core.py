import cupy as cp
import numpy as np

_BLOCK = 256

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
            # Fast path: single fused kernel (compute + carry-propagate + trim)
            smem = 3 * _BLOCK * 4  # 2 state buffers + 1 result buffer
            _addsub_small_kernel(
                (1,), (_BLOCK,),
                (a_gpu, len(a_gpu), b_gpu, len(b_gpu), result, n, extra, int(is_sub),
                 self._trim_idx_buf),
                shared_mem=smem)
            last = int(self._trim_idx_buf[0])  # single GPU→CPU sync
            if last < 0:
                return cp.array([0], dtype=cp.uint32)
            return result[:last + 1]
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
            return self._trim(result)

    def add(self, a_gpu, b_gpu):
        return self._addsub(a_gpu, b_gpu, False)

    def sub(self, a_gpu, b_gpu):
        return self._addsub(a_gpu, b_gpu, True)

    def mul(self, a_gpu, b_gpu):
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

        a_f = cp.zeros(n_fft, dtype=cp.float64)
        b_f = cp.zeros(n_fft, dtype=cp.float64)
        a_f[:n_a] = a_chunks.astype(cp.float64)
        b_f[:n_b] = b_chunks.astype(cp.float64)

        fa = cp.fft.rfft(a_f)
        fb = cp.fft.rfft(b_f)
        fa *= fb
        c = cp.fft.irfft(fa, n=n_fft)

        result = cp.zeros(n_fft + 1, dtype=cp.int64)
        result[:n_fft] = cp.rint(c).astype(cp.int64)

        chunk_mask = cp.int64((1 << chunk_bits) - 1)
        while True:
            carries = result >> cp.int64(chunk_bits)
            result &= chunk_mask
            if cp.all(carries == 0):
                break
            result[1:] += carries[:-1]

        # Recombine: view small-int array as uint32 limbs.
        out_dtype = cp.uint16 if chunk_bits == 16 else cp.uint8
        chunks_per_limb = 32 // chunk_bits  # 2 for uint16, 4 for uint8
        out = result.astype(out_dtype)
        pad = (-len(out)) % chunks_per_limb
        if pad:
            out = cp.concatenate([out, cp.zeros(pad, dtype=out_dtype)])
        return self._trim(out.view(cp.uint32))

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
        return self._trim(result)

    def _shift_right_one(self, a_gpu):
        n = len(a_gpu)
        result = cp.empty(n, dtype=cp.uint32)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _shift_right_one_kernel((blocks,), (_BLOCK,), (a_gpu, result, n))
        return self._trim(result)

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
