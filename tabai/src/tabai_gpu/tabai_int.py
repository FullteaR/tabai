from __future__ import annotations

import cupy as cp

from .core import GPUBigInt
from .utils import int_to_gpu, gpu_to_int

_shared_gpu_big_int = GPUBigInt()


class TabaiInt:
    def __init__(self, value: int | cp.ndarray):
        if isinstance(value, cp.ndarray):
            self._gpu = value
        else:
            self._gpu = int_to_gpu(value)

    def to_cpu(self) -> int:
        return gpu_to_int(self._gpu)

    def __add__(self, other: TabaiInt) -> TabaiInt:
        return TabaiInt(_shared_gpu_big_int.add(self._gpu, other._gpu))

    def __sub__(self, other: TabaiInt) -> TabaiInt:
        return TabaiInt(_shared_gpu_big_int.sub(self._gpu, other._gpu))

    def __mul__(self, other: TabaiInt) -> TabaiInt:
        return TabaiInt(_shared_gpu_big_int.mul(self._gpu, other._gpu))

    def __repr__(self) -> str:
        return f"TabaiInt({self.to_cpu()})"
