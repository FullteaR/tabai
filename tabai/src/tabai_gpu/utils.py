import cupy as cp
import numpy as np

def int_to_gpu(n: int) -> cp.ndarray:
    """Pythonの整数をGPU用Limb配列(uint32, Little-endian)に変換"""
    if n == 0:
        return cp.array([0], dtype=cp.uint32)
    
    # 16進数経由で変換 (巨大整数の場合、これが最も効率的)
    hex_s = hex(abs(n))[2:]
    # 8文字(32bit)単位にパディング
    pad_len = (8 - len(hex_s) % 8) % 8
    hex_s = hex_s.zfill(len(hex_s) + pad_len)
    
    limbs = [int(hex_s[i:i+8], 16) for i in range(0, len(hex_s), 8)]
    # Little-endianにするため逆転
    return cp.array(limbs[::-1], dtype=cp.uint32)

def gpu_to_int(gpu_arr: cp.ndarray) -> int:
    """GPU用Limb配列をPythonの整数に変換"""
    cpu_arr = cp.asnumpy(gpu_arr)
    out = 0
    for i, val in enumerate(cpu_arr):
        out |= int(val) << (i * 32)
    return out
