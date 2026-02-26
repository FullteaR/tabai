import cupy as cp

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
        if (states[n - 1] == 2 && is_sub == 0) {
            result[n] = 1u;
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



class GPUBigInt:
    def __init__(self, max_bits=1000000):
        self.max_bits = max_bits

    def _scan(self, states, n):
        blocks = (n + _BLOCK - 1) // _BLOCK
        smem = 2 * _BLOCK * 4

        if blocks == 1:
            dummy = cp.empty(1, dtype=cp.int32)
            _block_scan_kernel((1,), (_BLOCK,), (states, dummy, n), shared_mem=smem)
            return

        block_sums = cp.empty(blocks, dtype=cp.int32)
        _block_scan_kernel((blocks,), (_BLOCK,), (states, block_sums, n), shared_mem=smem)

        self._scan(block_sums, blocks)

        _propagate_kernel((blocks,), (_BLOCK,), (states, block_sums, n, _BLOCK))

    def _addsub(self, a_gpu, b_gpu, is_sub):
        n = max(len(a_gpu), len(b_gpu))
        extra = 0 if is_sub else 1
        result = cp.zeros(n + extra, dtype=cp.uint32)
        states = cp.empty(n, dtype=cp.int32)

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
        a16 = self._split_to_uint16(a_gpu)
        b16 = self._split_to_uint16(b_gpu)

        n_a = len(a16)
        n_b = len(b16)
        n_conv = n_a + n_b - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft <<= 1

        a_f = cp.zeros(n_fft, dtype=cp.float64)
        b_f = cp.zeros(n_fft, dtype=cp.float64)
        a_f[:n_a] = a16.astype(cp.float64)
        b_f[:n_b] = b16.astype(cp.float64)

        fa = cp.fft.rfft(a_f)
        fb = cp.fft.rfft(b_f)
        fa *= fb
        c = cp.fft.irfft(fa, n=n_fft)

        result = cp.zeros(n_fft + 1, dtype=cp.int64)
        result[:n_fft] = cp.rint(c).astype(cp.int64)

        while True:
            carries = result >> cp.int64(16)
            result = result & cp.int64(0xFFFF)
            if cp.all(carries == 0):
                break
            result[1:] += carries[:-1]

        return self._trim(self._combine_from_uint16(result.astype(cp.uint16)))

    def _split_to_uint16(self, arr):
        low = (arr & cp.uint32(0xFFFF)).astype(cp.uint16)
        high = (arr >> cp.uint32(16)).astype(cp.uint16)
        result = cp.empty(len(arr) * 2, dtype=cp.uint16)
        result[0::2] = low
        result[1::2] = high
        return result

    def _combine_from_uint16(self, arr):
        if len(arr) % 2 == 1:
            arr = cp.concatenate([arr, cp.array([0], dtype=cp.uint16)])
        low = arr[0::2].astype(cp.uint32)
        high = arr[1::2].astype(cp.uint32)
        return low | (high << cp.uint32(16))

    def _trim(self, gpu_arr):
        n = len(gpu_arr)
        if n == 0:
            return cp.array([0], dtype=cp.uint32)
        idx_buf = cp.full(1, -1, dtype=cp.int32)
        blocks = (n + _BLOCK - 1) // _BLOCK
        _find_last_nonzero_kernel((blocks,), (_BLOCK,), (gpu_arr, n, idx_buf))
        last = int(idx_buf[0])
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
        b_bits = self._bit_length(b)
        shift_max = a_bits - b_bits
        q_limbs = (shift_max + 32) // 32
        quotient = cp.zeros(q_limbs, dtype=cp.uint32)
        remainder = a.copy()
        shifted_b = self._shift_left(b, shift_max)
        for i in range(shift_max, -1, -1):
            if self._compare(remainder, shifted_b) >= 0:
                remainder = self.sub(remainder, shifted_b)
                limb_idx = i // 32
                bit_idx = i % 32
                quotient[limb_idx] = cp.uint32(int(quotient[limb_idx]) | (1 << bit_idx))
            if i > 0:
                shifted_b = self._shift_right_one(shifted_b)
        return self._trim(quotient), self._trim(remainder)

    def floordiv(self, a_gpu, b_gpu):
        q, _ = self.divmod(a_gpu, b_gpu)
        return q

    def mod(self, a_gpu, b_gpu):
        _, r = self.divmod(a_gpu, b_gpu)
        return r
