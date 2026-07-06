from __future__ import annotations

import cupy as cp

from .core import GPUBigInt
from .utils import int_to_gpu, gpu_to_int

_shared_gpu_big_int = GPUBigInt()
_ONE = cp.array([1], dtype=cp.uint32)


def _arr_is_zero(gpu: cp.ndarray) -> bool:
    # For a trimmed magnitude, len > 1 is always non-zero, so the (syncing)
    # single-element read only fires for a lone limb.
    return len(gpu) == 1 and int(gpu[0]) == 0


def _mag_cmp(a_gpu: cp.ndarray, b_gpu: cp.ndarray) -> int:
    """Compare two *trimmed* magnitudes.  Because a trimmed value's top limb is
    non-zero, a limb-count mismatch alone decides the comparison — skipping the
    compare-kernel launch/sync.  Equal lengths fall back to the engine compare.
    (Engine ``_compare`` also has untrimmed callers, so this shortcut lives only
    at the TabaiInt layer where the trimmed invariant always holds.)"""
    la, lb = len(a_gpu), len(b_gpu)
    if la != lb:
        return 1 if la > lb else -1
    return _shared_gpu_big_int._compare(a_gpu, b_gpu)


class TabaiInt:
    def __init__(self, value: int | cp.ndarray, sign: int = 1, *,
                 _trimmed: bool = False, _zero: bool | None = None):
        if isinstance(value, cp.ndarray):
            if _trimmed:
                self._gpu = value
                self._zero = _zero            # None => compute lazily on demand
            else:
                # Trimming already yields zero-ness for free.
                self._gpu, self._zero = _shared_gpu_big_int._trim_z(value)
            self._sign = sign
        else:
            self._sign = -1 if value < 0 else 1
            self._gpu = int_to_gpu(value)
            self._zero = (value == 0)
        # Zero's canonical sign is +1.  A +sign value is already canonical, so
        # only a negative sign can need flipping — letting positive results skip
        # the (possibly syncing) zero test entirely.
        if self._sign != 1 and self._is_zero():
            self._sign = 1

    def _is_zero(self) -> bool:
        z = self._zero
        if z is None:
            z = _arr_is_zero(self._gpu)
            self._zero = z
        return z

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
        # Magnitude arrays are immutable after construction (every kernel writes
        # a fresh output or engine-owned scratch, never an operand), so the
        # sign-flipped view can safely alias self._gpu without a copy.
        if self._is_zero():
            return TabaiInt(self._gpu, 1, _trimmed=True, _zero=True)
        return TabaiInt(self._gpu, -self._sign, _trimmed=True, _zero=False)

    def __abs__(self) -> TabaiInt:
        # Same immutability invariant as __neg__: share, don't copy.
        return TabaiInt(self._gpu, 1, _trimmed=True, _zero=self._zero)

    def __add__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        if self._sign == other._sign:
            result, z = _shared_gpu_big_int.add_trimmed(self._gpu, other._gpu)
            return TabaiInt(result, self._sign, _trimmed=True, _zero=z)
        cmp = _mag_cmp(self._gpu, other._gpu)
        if cmp == 0:
            return TabaiInt(0)
        if cmp > 0:
            result, z = _shared_gpu_big_int.sub_trimmed(self._gpu, other._gpu)
            return TabaiInt(result, self._sign, _trimmed=True, _zero=z)
        result, z = _shared_gpu_big_int.sub_trimmed(other._gpu, self._gpu)
        return TabaiInt(result, other._sign, _trimmed=True, _zero=z)

    def __sub__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        # Equivalent to self + (-other), but inlined so other._gpu is not copied.
        if self._sign != other._sign:
            result, z = _shared_gpu_big_int.add_trimmed(self._gpu, other._gpu)
            return TabaiInt(result, self._sign, _trimmed=True, _zero=z)
        cmp = _mag_cmp(self._gpu, other._gpu)
        if cmp == 0:
            return TabaiInt(0)
        if cmp > 0:
            result, z = _shared_gpu_big_int.sub_trimmed(self._gpu, other._gpu)
            return TabaiInt(result, self._sign, _trimmed=True, _zero=z)
        result, z = _shared_gpu_big_int.sub_trimmed(other._gpu, self._gpu)
        return TabaiInt(result, -other._sign, _trimmed=True, _zero=z)

    def __mul__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        if self._is_zero() or other._is_zero():
            return TabaiInt(0)
        result = _shared_gpu_big_int.mul(self._gpu, other._gpu)
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
        # q_gpu, r_gpu are trimmed magnitudes from the engine.
        if _arr_is_zero(r_gpu):
            q_zero = _arr_is_zero(q_gpu)
            sign = 1 if q_zero else self._sign * other._sign
            return TabaiInt(q_gpu, sign, _trimmed=True, _zero=q_zero), TabaiInt(0)
        if self._sign == other._sign:
            # r is non-zero here; q may still be zero (dividend < divisor), so
            # leave q's zero flag lazy rather than asserting it.
            return (TabaiInt(q_gpu, 1, _trimmed=True),
                    TabaiInt(r_gpu, self._sign, _trimmed=True, _zero=False))
        q_adj = _shared_gpu_big_int.add(q_gpu, _ONE)
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

    def __pow__(self, other: TabaiInt | int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        if other._sign == -1:
            raise ValueError("negative exponent is not supported")
        exp = gpu_to_int(other._gpu)
        if exp == 0:
            return TabaiInt(1)
        if exp == 1:
            return TabaiInt(self._gpu, self._sign, _trimmed=True, _zero=self._zero)
        calc = _shared_gpu_big_int
        base_gpu = self._gpu
        n_bits = exp.bit_length()

        # Sliding-window exponent size.  k=1 collapses to plain left-to-right
        # binary; larger k amortises more squarings per mul at the cost of
        # 2^(k-1)-1 precompute muls.  Thresholds chosen near the breakeven
        # 2^(k-1) ≈ n_bits/(k+1) - n_bits/(k+2).
        if n_bits <= 8:
            k = 1
        elif n_bits <= 24:
            k = 2
        elif n_bits <= 70:
            k = 3
        elif n_bits <= 196:
            k = 4
        elif n_bits <= 540:
            k = 5
        else:
            k = 6

        if k == 1:
            odd_powers = [base_gpu]
        else:
            base_sq = calc.mul(base_gpu, base_gpu)
            odd_powers = [base_gpu]
            for _ in range((1 << (k - 1)) - 1):
                odd_powers.append(calc.mul(odd_powers[-1], base_sq))

        # Left-to-right scan with sliding window.  The MSB is always 1, so the
        # first iteration seeds result_gpu via odd_powers (skipping the wasteful
        # "result=1, then mul by base" of the right-to-left form).
        i = n_bits - 1
        result_gpu = None
        while i >= 0:
            if not (exp >> i) & 1:
                result_gpu = calc.mul(result_gpu, result_gpu)
                i -= 1
                continue
            lo = max(0, i - k + 1)
            j = lo
            while not (exp >> j) & 1:
                j += 1
            win_bits = i - j + 1
            v = (exp >> j) & ((1 << win_bits) - 1)
            if result_gpu is None:
                result_gpu = odd_powers[(v - 1) >> 1]
            else:
                for _ in range(win_bits):
                    result_gpu = calc.mul(result_gpu, result_gpu)
                result_gpu = calc.mul(result_gpu, odd_powers[(v - 1) >> 1])
            i = j - 1

        result_sign = self._sign if exp & 1 else 1
        return TabaiInt(result_gpu, result_sign)

    def __rpow__(self, other: int) -> TabaiInt:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other.__pow__(self)

    def _cmp(self, other: TabaiInt) -> int:
        if self._sign != other._sign:
            return 1 if self._sign > other._sign else -1
        cmp = _mag_cmp(self._gpu, other._gpu)
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
