import cupy as cp
import numpy as np


def int_to_gpu(n: int) -> cp.ndarray:
    if n == 0:
        return cp.array([0], dtype=cp.uint32)
    abs_n = abs(n)
    byte_len = (abs_n.bit_length() + 7) // 8
    b = abs_n.to_bytes(byte_len, byteorder='little')
    pad = (-len(b)) % 4
    if pad:
        b = b + b'\x00' * pad
    np_arr = np.frombuffer(b, dtype=np.uint32).copy()
    return cp.asarray(np_arr)


def gpu_to_int(gpu_arr: cp.ndarray) -> int:
    cpu_arr = cp.asnumpy(gpu_arr).astype(np.uint32)
    return int.from_bytes(cpu_arr.tobytes(), byteorder='little')
