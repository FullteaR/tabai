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

_mul_kernel = cp.RawKernel(r'''
extern "C" __global__
void mul_partial_products(const unsigned short* a, int n_a,
                          const unsigned short* b, int n_b,
                          unsigned long long* result, int n_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_out) return;

    unsigned long long sum = 0;
    int i_start = (k - n_b + 1) > 0 ? (k - n_b + 1) : 0;
    int i_end = k < (n_a - 1) ? k : (n_a - 1);

    for (int i = i_start; i <= i_end; i++) {
        sum += (unsigned long long)a[i] * (unsigned long long)b[k - i];
    }

    result[k] = sum;
}
''', 'mul_partial_products')

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
        """GPU並列乗算"""
        a16 = self._split_to_uint16(a_gpu)
        b16 = self._split_to_uint16(b_gpu)

        n_a = len(a16)
        n_b = len(b16)
        n_out = n_a + n_b

        result = cp.zeros(n_out, dtype=cp.uint64)

        threads_per_block = 256
        blocks = (n_out + threads_per_block - 1) // threads_per_block
        _mul_kernel((blocks,), (threads_per_block,),
                    (a16, n_a, b16, n_b, result, n_out))
        cp.cuda.runtime.deviceSynchronize()

        while True:
            carries = result >> cp.uint64(16)
            result = result & cp.uint64(0xFFFF)
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
