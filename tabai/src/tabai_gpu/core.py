import cupy as cp
from .utils import int_to_gpu, gpu_to_int

# Hillis-Steele アルゴリズムによる並列スキャンカーネル
# 状態: 0=Kill, 1=Propagate, 2=Generate
# ロジック: 右の要素がPropagate(1)なら左の要素の状態を引き継ぐ
_parallel_scan_kernel = cp.RawKernel(r'''
extern "C" __global__
void parallel_carry_scan(int* states, int n, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int prev_idx = idx - step;
    if (prev_idx >= 0) {
        int current = states[idx];
        int prev = states[prev_idx];
        
        // 自分の状態がPropagate(1)なら、step分前の状態を上書きする
        if (current == 1) {
            states[idx] = prev;
        }
    }
}
''', 'parallel_carry_scan')

class GPUBigInt:
    def __init__(self, max_bits=1000000):
        self.max_bits = max_bits

    def _resolve_carries_gpu(self, states):
        """CUDAカーネルを繰り返し呼び出して O(log N) でキャリーを確定させる"""
        n = states.size
        threads_per_block = 256
        blocks = (n + threads_per_block - 1) // threads_per_block
        
        # ステップ幅を 1, 2, 4, 8... と倍にしながらスキャン
        step = 1
        while step < n:
            _parallel_scan_kernel((blocks,), (threads_per_block,), (states, n, step))
            cp.cuda.runtime.deviceSynchronize() # 各ステップの完了を待機
            step *= 2
        return states

    def _align_and_get_states(self, a_gpu, b_gpu, mode='add'):
        n = max(len(a_gpu), len(b_gpu))
        a = cp.zeros(n, dtype=cp.uint32)
        b = cp.zeros(n, dtype=cp.uint32)
        a[:len(a_gpu)] = a_gpu
        b[:len(b_gpu)] = b_gpu

        if mode == 'add':
            sum64 = a.astype(cp.uint64) + b.astype(cp.uint64)
            states = cp.where(sum64 > 0xFFFFFFFF, 2, 
                             cp.where(sum64 == 0xFFFFFFFF, 1, 0)).astype(cp.int32)
            return states, sum64.astype(cp.uint32)
        else:
            diff64 = a.astype(cp.int64) - b.astype(cp.int64)
            states = cp.where(diff64 < 0, 2, 
                             cp.where(diff64 == 0, 1, 0)).astype(cp.int32)
            return states, diff64.astype(cp.uint32)

    def add(self, a_gpu, b_gpu):
        states, base_res = self._align_and_get_states(a_gpu, b_gpu, 'add')
        resolved = self._resolve_carries_gpu(states)
        
        actual_carries = cp.zeros(len(states), dtype=cp.uint32)
        if len(states) > 1:
            actual_carries[1:] = (resolved[:-1] == 2).astype(cp.uint32)
        
        res = base_res + actual_carries
        if resolved[-1] == 2:
            res = cp.concatenate([res, cp.array([1], dtype=cp.uint32)])
        return self._trim(res)

    def sub(self, a_gpu, b_gpu):
        states, base_res = self._align_and_get_states(a_gpu, b_gpu, 'sub')
        resolved = self._resolve_carries_gpu(states)
        
        actual_borrows = cp.zeros(len(states), dtype=cp.uint32)
        if len(states) > 1:
            actual_borrows[1:] = (resolved[:-1] == 2).astype(cp.uint32)
        
        res = base_res - actual_borrows
        return self._trim(res)

    def mul(self, a_gpu, b_gpu):
        """GPU FFT乗算"""
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

        fc = cp.fft.rfft(a_f) * cp.fft.rfft(b_f)
        c = cp.fft.irfft(fc, n=n_fft)

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
        """uint32 limb配列をuint16 half-limb配列に分割 (little-endian)"""
        low = (arr & cp.uint32(0xFFFF)).astype(cp.uint16)
        high = (arr >> cp.uint32(16)).astype(cp.uint16)
        result = cp.empty(len(arr) * 2, dtype=cp.uint16)
        result[0::2] = low
        result[1::2] = high
        return result

    def _combine_from_uint16(self, arr):
        """uint16 half-limb配列をuint32 limb配列に結合"""
        if len(arr) % 2 == 1:
            arr = cp.concatenate([arr, cp.array([0], dtype=cp.uint16)])
        low = arr[0::2].astype(cp.uint32)
        high = arr[1::2].astype(cp.uint32)
        return low | (high << cp.uint32(16))

    def _trim(self, gpu_arr):
        nonzero = cp.where(gpu_arr != 0)[0]
        if nonzero.size == 0:
            return cp.array([0], dtype=cp.uint32)
        return gpu_arr[:int(nonzero[-1]) + 1]

    def _compare(self, a_gpu, b_gpu):
        n = max(len(a_gpu), len(b_gpu))
        a = cp.zeros(n, dtype=cp.uint32)
        b = cp.zeros(n, dtype=cp.uint32)
        a[:len(a_gpu)] = a_gpu
        b[:len(b_gpu)] = b_gpu
        diff = a.astype(cp.int64) - b.astype(cp.int64)
        nonzero_idx = cp.nonzero(diff)[0]
        if len(nonzero_idx) == 0:
            return 0
        top_idx = int(nonzero_idx[-1])
        return 1 if int(diff[top_idx]) > 0 else -1

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
        if bit_shift == 0:
            result[limb_shift:limb_shift + len(a_gpu)] = a_gpu
        else:
            ext = a_gpu.astype(cp.uint64) << cp.uint64(bit_shift)
            result[limb_shift:limb_shift + len(a_gpu)] = (ext & 0xFFFFFFFF).astype(cp.uint32)
            result[limb_shift + 1:limb_shift + 1 + len(a_gpu)] += (ext >> 32).astype(cp.uint32)
        return self._trim(result)

    def _shift_right_one(self, a_gpu):
        result = a_gpu >> cp.uint32(1)
        if len(a_gpu) > 1:
            carry_bits = (a_gpu[1:] & cp.uint32(1)) << cp.uint32(31)
            result[:-1] |= carry_bits
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
