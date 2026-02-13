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

    @staticmethod
    def _coerce(other: TabaiInt | int) -> TabaiInt:
        if isinstance(other, TabaiInt):
            return other
        if isinstance(other, int):
            return TabaiInt(other)
        return NotImplemented

    def to_cpu(self) -> int:
        val = gpu_to_int(self._gpu)
        return val if self._sign == 1 else -val

    def __neg__(self) -> TabaiInt:
        if _is_zero(self._gpu):
            return TabaiInt(self._gpu.copy(), 1)
        return TabaiInt(self._gpu.copy(), -self._sign)

    def __abs__(self) -> TabaiInt:
        return TabaiInt(self._gpu.copy(), 1)

    def __add__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
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

    def __sub__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __mul__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        result = _shared_gpu_big_int.mul(self._gpu, other._gpu)
        if _is_zero(result):
            return TabaiInt(result, 1)
        return TabaiInt(result, self._sign * other._sign)

    def __floordiv__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        _, r = divmod(self, other)
        return r

    def __divmod__(self, other: TabaiInt | int) -> tuple[TabaiInt, TabaiInt]:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
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

    def __radd__(self, other: int) -> TabaiInt:
        return self.__add__(other)

    def __rsub__(self, other: int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__sub__(self)

    def __rmul__(self, other: int) -> TabaiInt:
        return self.__mul__(other)

    def __rfloordiv__(self, other: int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__floordiv__(self)

    def __rmod__(self, other: int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__mod__(self)

    def __rdivmod__(self, other: int) -> tuple[TabaiInt, TabaiInt]:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__divmod__(self)

    def _cmp(self, other: TabaiInt) -> int:
        if self._sign != other._sign:
            return 1 if self._sign > other._sign else -1
        cmp = _shared_gpu_big_int._compare(self._gpu, other._gpu)
        if self._sign == -1:
            cmp = -cmp
        return cmp

    def __eq__(self, other: object) -> bool:
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self._cmp(coerced) == 0

    def __ne__(self, other: object) -> bool:
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self._cmp(coerced) != 0

    def __lt__(self, other: TabaiInt | int) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self._cmp(other) < 0

    def __le__(self, other: TabaiInt | int) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self._cmp(other) <= 0

    def __gt__(self, other: TabaiInt | int) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self._cmp(other) > 0

    def __ge__(self, other: TabaiInt | int) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self._cmp(other) >= 0

    def __repr__(self) -> str:
        return f"TabaiInt({self.to_cpu()})"
