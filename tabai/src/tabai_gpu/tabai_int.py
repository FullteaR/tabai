from __future__ import annotations

import cupy as cp

from .core import GPUBigInt
from .utils import int_to_gpu, gpu_to_int

_shared_gpu_big_int = GPUBigInt()


def _is_zero(gpu: cp.ndarray) -> bool:
    return len(gpu) == 1 and int(gpu[0]) == 0


class TabaiInt:
    def __init__(self, value: int | cp.ndarray, sign: int = 1):
        if isinstance(value, cp.ndarray):
            self._gpu = value
            self._sign = sign
        else:
            self._sign = -1 if value < 0 else 1
            self._gpu = int_to_gpu(value)
        if _is_zero(self._gpu):
            self._sign = 1

    def to_cpu(self) -> int:
        val = gpu_to_int(self._gpu)
        return val if self._sign == 1 else -val

    def __neg__(self) -> TabaiInt:
        if _is_zero(self._gpu):
            return TabaiInt(self._gpu.copy(), 1)
        return TabaiInt(self._gpu.copy(), -self._sign)

    def __abs__(self) -> TabaiInt:
        return TabaiInt(self._gpu.copy(), 1)

    def __add__(self, other: TabaiInt) -> TabaiInt:
        if self._sign == other._sign:
            result = _shared_gpu_big_int.add(self._gpu, other._gpu)
            return TabaiInt(result, self._sign)
        cmp = _shared_gpu_big_int._compare(self._gpu, other._gpu)
        if cmp == 0:
            return TabaiInt(0)
        if cmp > 0:
            result = _shared_gpu_big_int.sub(self._gpu, other._gpu)
            return TabaiInt(result, self._sign)
        result = _shared_gpu_big_int.sub(other._gpu, self._gpu)
        return TabaiInt(result, other._sign)

    def __sub__(self, other: TabaiInt) -> TabaiInt:
        return self + (-other)

    def __mul__(self, other: TabaiInt) -> TabaiInt:
        result = _shared_gpu_big_int.mul(self._gpu, other._gpu)
        if _is_zero(result):
            return TabaiInt(result, 1)
        return TabaiInt(result, self._sign * other._sign)

    def __floordiv__(self, other: TabaiInt) -> TabaiInt:
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other: TabaiInt) -> TabaiInt:
        _, r = divmod(self, other)
        return r

    def __divmod__(self, other: TabaiInt) -> tuple[TabaiInt, TabaiInt]:
        q_gpu, r_gpu = _shared_gpu_big_int.divmod(self._gpu, other._gpu)
        if _is_zero(r_gpu):
            sign = 1 if _is_zero(q_gpu) else self._sign * other._sign
            return TabaiInt(q_gpu, sign), TabaiInt(0)
        if self._sign == other._sign:
            return TabaiInt(q_gpu, 1), TabaiInt(r_gpu, self._sign)
        one = cp.array([1], dtype=cp.uint32)
        q_adj = _shared_gpu_big_int.add(q_gpu, one)
        r_adj = _shared_gpu_big_int.sub(other._gpu, r_gpu)
        return TabaiInt(q_adj, -1), TabaiInt(r_adj, other._sign)

    def _cmp(self, other: TabaiInt) -> int:
        if self._sign != other._sign:
            return 1 if self._sign > other._sign else -1
        cmp = _shared_gpu_big_int._compare(self._gpu, other._gpu)
        if self._sign == -1:
            cmp = -cmp
        return cmp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TabaiInt):
            return NotImplemented
        return self._cmp(other) == 0

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, TabaiInt):
            return NotImplemented
        return self._cmp(other) != 0

    def __lt__(self, other: TabaiInt) -> bool:
        return self._cmp(other) < 0

    def __le__(self, other: TabaiInt) -> bool:
        return self._cmp(other) <= 0

    def __gt__(self, other: TabaiInt) -> bool:
        return self._cmp(other) > 0

    def __ge__(self, other: TabaiInt) -> bool:
        return self._cmp(other) >= 0

    def __repr__(self) -> str:
        return f"TabaiInt({self.to_cpu()})"
