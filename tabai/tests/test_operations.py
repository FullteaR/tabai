import pytest
import random
import cupy as cp
from tabai_gpu.core import GPUBigInt
from tabai_gpu.utils import int_to_gpu, gpu_to_int
from tabai_gpu.tabai_int import TabaiInt

import sys
sys.set_int_max_str_digits(0)

@pytest.fixture
def calc():
    return GPUBigInt()

def test_add_1_1(calc):
    a, b = 1, 1
    res = calc.add(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == 2

def test_add_basic(calc):
    a, b = 10**100, 20**100
    res = calc.add(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a + b

def test_add_small_numbers(calc):
    for a in range(0,100):
        for b in range(0,100):
            res = calc.add(int_to_gpu(a), int_to_gpu(b))
            assert gpu_to_int(res) == a + b

def test_add_carry_chain(calc):
    a = (1 << 100000) - 1
    b = 1
    res = calc.add(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == (1 << 100000)

def test_sub_basic(calc):
    a, b = 20**100, 10**100
    res = calc.sub(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a - b

def test_mul_1_1(calc):
    res = calc.mul(int_to_gpu(1), int_to_gpu(1))
    assert gpu_to_int(res) == 1

def test_mul_basic(calc):
    a, b = 10**100, 20**100
    res = calc.mul(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a * b

def test_mul_by_zero(calc):
    a = 10**100
    res = calc.mul(int_to_gpu(a), int_to_gpu(0))
    assert gpu_to_int(res) == 0

def test_mul_by_one(calc):
    a = 10**100
    res = calc.mul(int_to_gpu(a), int_to_gpu(1))
    assert gpu_to_int(res) == a

def test_mul_small_numbers(calc):
    for a in range(0,100):
        for b in range(0,100):
            res = calc.mul(int_to_gpu(a), int_to_gpu(b))
            assert gpu_to_int(res) == a * b

def test_mul_carry_chain(calc):
    a = (1 << 100000) - 1
    b = (1 << 100000) - 1
    res = calc.mul(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a * b

@pytest.mark.parametrize("a,b", [
    ((1 << 1000) - (1 << 500) + 1, (1 << 999) + (1 << 333) - 1),
    ((1 << 100000) - (1 << 50000) + 1, (1 << 99999) + (1 << 33333) - 1),
], ids=["1000bit", "100000bit"])
def test_large_ops(calc, a, b):
    assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b
    assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b
    assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b


# ---------------------------------------------------------------------------
# Boundary tests for the fused small-kernel path (n <= 256 limbs = 8192 bits)
# ---------------------------------------------------------------------------

# The fused kernel handles n <= _BLOCK=256 limbs.
# These bit widths straddle the boundary.
_SMALL_BITS = 8192   # 256 limbs — last size handled by the fused kernel
_LARGE_BITS = 8193   # 257 limbs — first size that falls through to the multi-kernel path


class TestAddSubBoundary:
    """Verify correctness at and around the 256-limb fused-kernel boundary."""

    # --- small-kernel side (n <= 256) ---

    def test_add_at_limit(self, calc):
        a = (1 << (_SMALL_BITS - 1))
        b = (1 << (_SMALL_BITS - 2))
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    def test_add_carry_chain_fills_256_limbs(self, calc):
        # All 256 limbs are 0xFFFFFFFF; adding 1 must carry through every limb.
        a = (1 << _SMALL_BITS) - 1
        b = 1
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == 1 << _SMALL_BITS

    def test_add_carry_out_creates_257th_limb(self, calc):
        # result needs an extra limb beyond 256
        a = (1 << _SMALL_BITS) - 1
        b = (1 << _SMALL_BITS) - 1
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    def test_sub_at_limit(self, calc):
        a = (1 << _SMALL_BITS) - 1
        b = (1 << (_SMALL_BITS // 2))
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b

    def test_sub_borrow_chain_fills_256_limbs(self, calc):
        # 2^8192 - 1: borrow propagates across all limbs.
        a = 1 << _SMALL_BITS
        b = 1
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b

    def test_add_asymmetric_sizes_small_path(self, calc):
        # len(a)=256 limbs, len(b)=1 limb → n=256 (fused kernel)
        a = (1 << (_SMALL_BITS - 1))
        b = 1
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    # --- large-kernel side (n > 256) ---

    def test_add_just_above_limit(self, calc):
        a = 1 << _LARGE_BITS
        b = 1
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    def test_add_carry_chain_257_limbs(self, calc):
        a = (1 << _LARGE_BITS) - 1
        b = 1
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == 1 << _LARGE_BITS

    def test_sub_just_above_limit(self, calc):
        a = (1 << _LARGE_BITS) - 1
        b = (1 << (_LARGE_BITS // 2))
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b

    # --- crossing the boundary (one operand each side) ---

    @pytest.mark.parametrize("a_bits,b_bits", [
        (_SMALL_BITS, 1),          # 256-limb + 1-limb  → n=256 (fused)
        (_LARGE_BITS, 1),          # 257-limb + 1-limb  → n=257 (multi)
        (_LARGE_BITS, _SMALL_BITS),# 257-limb + 256-limb → n=257 (multi)
    ])
    def test_add_mixed_sizes(self, calc, a_bits, b_bits):
        a = (1 << (a_bits - 1)) | ((1 << (a_bits // 3)) - 1)
        b = (1 << (b_bits - 1)) | 1
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    @pytest.mark.parametrize("bits", [
        _SMALL_BITS - 1,  # well inside fused path
        _SMALL_BITS,      # exactly at boundary (fused)
        _SMALL_BITS + 1,  # just outside (multi)
        _SMALL_BITS + 32, # one full limb past boundary (multi)
    ])
    def test_add_random_around_boundary(self, calc, bits):
        random.seed(bits)
        a = random.getrandbits(bits) | (1 << (bits - 1))
        b = random.getrandbits(bits) | (1 << (bits - 1))
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    @pytest.mark.parametrize("bits", [
        _SMALL_BITS - 1,
        _SMALL_BITS,
        _SMALL_BITS + 1,
        _SMALL_BITS + 32,
    ])
    def test_sub_random_around_boundary(self, calc, bits):
        random.seed(bits)
        a = random.getrandbits(bits) | (1 << bits)  # ensure a > b
        b = random.getrandbits(bits) | (1 << (bits - 1))
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b


class TestNegativeNumbers:
    def test_neg_construction(self):
        assert TabaiInt(-1).to_cpu() == -1
        assert TabaiInt(-100).to_cpu() == -100
        assert TabaiInt(-10**100).to_cpu() == -(10**100)
        assert TabaiInt(0).to_cpu() == 0

    def test_neg_operator(self):
        assert (-TabaiInt(5)).to_cpu() == -5
        assert (-TabaiInt(-5)).to_cpu() == 5
        assert (-TabaiInt(0)).to_cpu() == 0

    def test_abs_operator(self):
        assert abs(TabaiInt(-5)).to_cpu() == 5
        assert abs(TabaiInt(5)).to_cpu() == 5
        assert abs(TabaiInt(0)).to_cpu() == 0

    def test_add_neg_neg(self):
        a, b = -3, -7
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_add_neg_pos(self):
        assert (TabaiInt(-3) + TabaiInt(7)).to_cpu() == 4
        assert (TabaiInt(-7) + TabaiInt(3)).to_cpu() == -4

    def test_add_pos_neg(self):
        assert (TabaiInt(7) + TabaiInt(-3)).to_cpu() == 4
        assert (TabaiInt(3) + TabaiInt(-7)).to_cpu() == -4

    def test_add_cancel_to_zero(self):
        assert (TabaiInt(5) + TabaiInt(-5)).to_cpu() == 0
        assert (TabaiInt(-5) + TabaiInt(5)).to_cpu() == 0

    def test_sub_neg(self):
        assert (TabaiInt(-3) - TabaiInt(7)).to_cpu() == -10
        assert (TabaiInt(3) - TabaiInt(-7)).to_cpu() == 10
        assert (TabaiInt(-3) - TabaiInt(-7)).to_cpu() == 4

    def test_mul_neg(self):
        assert (TabaiInt(-3) * TabaiInt(7)).to_cpu() == -21
        assert (TabaiInt(3) * TabaiInt(-7)).to_cpu() == -21
        assert (TabaiInt(-3) * TabaiInt(-7)).to_cpu() == 21
        assert (TabaiInt(-3) * TabaiInt(0)).to_cpu() == 0
        assert (TabaiInt(0) * TabaiInt(-7)).to_cpu() == 0

    def test_add_neg_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_sub_neg_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b

    def test_mul_neg_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_neg_large(self):
        a = -(10**100)
        b = 20**100
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_floordiv_neg(self):
        assert (TabaiInt(-7) // TabaiInt(2)).to_cpu() == -7 // 2
        assert (TabaiInt(7) // TabaiInt(-2)).to_cpu() == 7 // -2
        assert (TabaiInt(-7) // TabaiInt(-2)).to_cpu() == -7 // -2
        assert (TabaiInt(-6) // TabaiInt(2)).to_cpu() == -6 // 2

    def test_mod_neg(self):
        assert (TabaiInt(-7) % TabaiInt(2)).to_cpu() == -7 % 2
        assert (TabaiInt(7) % TabaiInt(-2)).to_cpu() == 7 % -2
        assert (TabaiInt(-7) % TabaiInt(-2)).to_cpu() == -7 % -2
        assert (TabaiInt(-6) % TabaiInt(2)).to_cpu() == -6 % 2

    def test_divmod_neg(self):
        for a in [-7, 7, -6, 6, -100, 100]:
            for b in [-3, 3, -2, 2, -1, 1]:
                q, r = divmod(TabaiInt(a), TabaiInt(b))
                eq, er = divmod(a, b)
                assert q.to_cpu() == eq
                assert r.to_cpu() == er

    def test_repr_neg(self):
        assert repr(TabaiInt(-42)) == "TabaiInt(-42)"
        assert repr(TabaiInt(42)) == "TabaiInt(42)"
        assert repr(TabaiInt(0)) == "TabaiInt(0)"


class TestComparison:
    def test_eq(self):
        assert TabaiInt(5) == TabaiInt(5)
        assert TabaiInt(-5) == TabaiInt(-5)
        assert TabaiInt(0) == TabaiInt(0)
        assert not (TabaiInt(5) == TabaiInt(-5))
        assert not (TabaiInt(5) == TabaiInt(3))

    def test_ne(self):
        assert TabaiInt(5) != TabaiInt(3)
        assert TabaiInt(5) != TabaiInt(-5)
        assert not (TabaiInt(5) != TabaiInt(5))

    def test_lt(self):
        assert TabaiInt(-5) < TabaiInt(3)
        assert TabaiInt(-5) < TabaiInt(-3)
        assert TabaiInt(3) < TabaiInt(5)
        assert not (TabaiInt(5) < TabaiInt(3))
        assert not (TabaiInt(5) < TabaiInt(5))

    def test_le(self):
        assert TabaiInt(-5) <= TabaiInt(3)
        assert TabaiInt(5) <= TabaiInt(5)
        assert not (TabaiInt(5) <= TabaiInt(3))

    def test_gt(self):
        assert TabaiInt(5) > TabaiInt(3)
        assert TabaiInt(-3) > TabaiInt(-5)
        assert TabaiInt(3) > TabaiInt(-5)
        assert not (TabaiInt(3) > TabaiInt(5))
        assert not (TabaiInt(5) > TabaiInt(5))

    def test_ge(self):
        assert TabaiInt(5) >= TabaiInt(3)
        assert TabaiInt(5) >= TabaiInt(5)
        assert not (TabaiInt(3) >= TabaiInt(5))

    def test_compare_zero(self):
        assert TabaiInt(0) == TabaiInt(0)
        assert TabaiInt(0) >= TabaiInt(0)
        assert TabaiInt(0) <= TabaiInt(0)
        assert TabaiInt(1) > TabaiInt(0)
        assert TabaiInt(-1) < TabaiInt(0)
        assert TabaiInt(0) > TabaiInt(-1)
        assert TabaiInt(0) < TabaiInt(1)

    def test_compare_large(self):
        a = 10**100
        b = 20**100
        assert TabaiInt(a) < TabaiInt(b)
        assert TabaiInt(-b) < TabaiInt(-a)
        assert TabaiInt(-a) > TabaiInt(-b)
        assert TabaiInt(-b) < TabaiInt(a)

    def test_compare_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                ta, tb = TabaiInt(a), TabaiInt(b)
                assert (ta == tb) == (a == b)
                assert (ta != tb) == (a != b)
                assert (ta < tb) == (a < b)
                assert (ta <= tb) == (a <= b)
                assert (ta > tb) == (a > b)
                assert (ta >= tb) == (a >= b)

    def test_compare_random_large(self):
        random.seed(42)
        for _ in range(50):
            a = random.randint(-(1 << 1000), 1 << 1000)
            b = random.randint(-(1 << 1000), 1 << 1000)
            ta, tb = TabaiInt(a), TabaiInt(b)
            assert (ta == tb) == (a == b)
            assert (ta < tb) == (a < b)
            assert (ta > tb) == (a > b)


class TestIntInterop:
    def test_add_tabai_int(self):
        assert (TabaiInt(10) + 3).to_cpu() == 13
        assert (TabaiInt(-10) + 3).to_cpu() == -7
        assert (TabaiInt(10) + (-3)).to_cpu() == 7

    def test_radd_int_tabai(self):
        assert (3 + TabaiInt(10)).to_cpu() == 13
        assert (3 + TabaiInt(-10)).to_cpu() == -7
        assert ((-3) + TabaiInt(10)).to_cpu() == 7

    def test_sub_tabai_int(self):
        assert (TabaiInt(10) - 3).to_cpu() == 7
        assert (TabaiInt(-10) - 3).to_cpu() == -13
        assert (TabaiInt(10) - (-3)).to_cpu() == 13

    def test_rsub_int_tabai(self):
        assert (3 - TabaiInt(10)).to_cpu() == -7
        assert (3 - TabaiInt(-10)).to_cpu() == 13
        assert ((-3) - TabaiInt(10)).to_cpu() == -13

    def test_mul_tabai_int(self):
        assert (TabaiInt(10) * 3).to_cpu() == 30
        assert (TabaiInt(-10) * 3).to_cpu() == -30
        assert (TabaiInt(10) * (-3)).to_cpu() == -30
        assert (TabaiInt(10) * 0).to_cpu() == 0

    def test_rmul_int_tabai(self):
        assert (3 * TabaiInt(10)).to_cpu() == 30
        assert (3 * TabaiInt(-10)).to_cpu() == -30
        assert ((-3) * TabaiInt(10)).to_cpu() == -30
        assert (0 * TabaiInt(10)).to_cpu() == 0

    def test_floordiv_tabai_int(self):
        assert (TabaiInt(10) // 3).to_cpu() == 3
        assert (TabaiInt(-7) // 2).to_cpu() == -7 // 2
        assert (TabaiInt(7) // (-2)).to_cpu() == 7 // -2

    def test_rfloordiv_int_tabai(self):
        assert (10 // TabaiInt(3)).to_cpu() == 3
        assert ((-7) // TabaiInt(2)).to_cpu() == -7 // 2
        assert (7 // TabaiInt(-2)).to_cpu() == 7 // -2

    def test_mod_tabai_int(self):
        assert (TabaiInt(10) % 3).to_cpu() == 1
        assert (TabaiInt(-7) % 2).to_cpu() == -7 % 2
        assert (TabaiInt(7) % (-2)).to_cpu() == 7 % -2

    def test_rmod_int_tabai(self):
        assert (10 % TabaiInt(3)).to_cpu() == 1
        assert ((-7) % TabaiInt(2)).to_cpu() == -7 % 2
        assert (7 % TabaiInt(-2)).to_cpu() == 7 % -2

    def test_divmod_tabai_int(self):
        q, r = divmod(TabaiInt(10), 3)
        assert q.to_cpu() == 3
        assert r.to_cpu() == 1

    def test_rdivmod_int_tabai(self):
        q, r = divmod(10, TabaiInt(3))
        assert q.to_cpu() == 3
        assert r.to_cpu() == 1

    def test_compare_tabai_int(self):
        assert TabaiInt(5) == 5
        assert TabaiInt(5) != 3
        assert TabaiInt(5) > 3
        assert TabaiInt(5) >= 5
        assert TabaiInt(3) < 5
        assert TabaiInt(3) <= 3

    def test_int_interop_exhaustive(self):
        for a in range(-20, 21):
            for b in range(-20, 21):
                ta = TabaiInt(a)
                assert (ta + b).to_cpu() == a + b
                assert (b + ta).to_cpu() == b + a
                assert (ta - b).to_cpu() == a - b
                assert (b - ta).to_cpu() == b - a
                assert (ta * b).to_cpu() == a * b
                assert (b * ta).to_cpu() == b * a
                if b != 0:
                    assert (ta // b).to_cpu() == a // b
                    assert (ta % b).to_cpu() == a % b
                if a != 0:
                    assert (b // ta).to_cpu() == b // a
                    assert (b % ta).to_cpu() == b % a

    def test_int_interop_large(self):
        a = 10**100
        b = 20**100
        assert (TabaiInt(a) + b).to_cpu() == a + b
        assert (b + TabaiInt(a)).to_cpu() == b + a
        assert (TabaiInt(a) - b).to_cpu() == a - b
        assert (b - TabaiInt(a)).to_cpu() == b - a
        assert (TabaiInt(a) * b).to_cpu() == a * b
        assert (b * TabaiInt(a)).to_cpu() == b * a


# ---------------------------------------------------------------------------
# Tests for the Phase 1+2+4 mul optimisation
# (fixed-count carry propagation kernel + pre-allocated FFT workspace)
#
# Correctness is verified with identities whose expected values can be
# computed cheaply without full Python big-int multiplication:
#
#   Identity P:  2^n * 2^m  =  2^(n+m)
#   Identity D:  (2^n + 1) * (2^n - 1)  =  2^(2n) - 1
#   Identity A:  a * 2  ==  a + a
#
# Carry-stress test:  (2^n - 1) * (2^n - 1)
#   All uint16/uint8 chunks are at their maximum value, producing the
#   largest possible FFT convolution coefficients and the deepest carry
#   chains — the hardest case for the fixed-count carry loop.
#
# Boundary sizes used:
#   _FFT_B16_BITS  — inside the B=16 path  (n_fft_est < 2^20)
#   _FFT_B8_BITS   — inside the B= 8 path  (n_fft_est ≥ 2^20)
# ---------------------------------------------------------------------------

# Above the schoolbook/FFT dispatch threshold so the FFT path is genuinely
# exercised.  Dispatch is on work = la*lb; a square power-of-2 product 1<<n *
# 1<<m stores each operand as ~bits/2/32 limbs, so bits must exceed ~2*5120*32
# (~327_680) for the pair's work to clear _MUL_SCHOOLBOOK_MAX_WORK = 5120*5120.
_FFT_ENTRY_BITS = 350_000  # ~5469 limbs/operand — smallest FFT-path size, B=16
_FFT_B16_BITS   = 500_000  # B=16 path, moderately sized FFT
_FFT_B8_BITS    = 5_000_000  # B=8 path  (n_fft_est ≥ 2^20)


class TestMulFFTCarryProp:
    """Correctness of the fixed-count carry propagation kernel (Phase 1+2)."""

    # -- Identity P: 2^n * 2^m = 2^(n+m) ----------------------------------

    @pytest.mark.parametrize("bits", [
        _FFT_ENTRY_BITS,
        _FFT_B16_BITS,
        _FFT_B8_BITS,
    ], ids=["entry", "b16", "b8"])
    def test_power_of_2_product(self, calc, bits):
        n = bits // 2
        m = bits - n
        result = gpu_to_int(calc.mul(int_to_gpu(1 << n), int_to_gpu(1 << m)))
        assert result == (1 << (n + m))

    # -- Identity D: (2^n + 1)(2^n - 1) = 2^(2n) - 1 ----------------------

    @pytest.mark.parametrize("bits", [
        _FFT_ENTRY_BITS,
        _FFT_B16_BITS,
        _FFT_B8_BITS,
    ], ids=["entry", "b16", "b8"])
    def test_diff_of_squares(self, calc, bits):
        n = bits // 2
        a_val = (1 << n) + 1
        b_val = (1 << n) - 1
        result = gpu_to_int(calc.mul(int_to_gpu(a_val), int_to_gpu(b_val)))
        assert result == (1 << (2 * n)) - 1

    # -- Identity A: a * 2 == a + a  (self-consistency) --------------------

    @pytest.mark.parametrize("bits", [
        _FFT_ENTRY_BITS,
        _FFT_B16_BITS,
        _FFT_B8_BITS,
    ], ids=["entry", "b16", "b8"])
    def test_mul2_equals_add(self, calc, bits):
        random.seed(bits)
        a_int = random.getrandbits(bits) | (1 << (bits - 1))
        a = int_to_gpu(a_int)
        result_mul = gpu_to_int(calc.mul(a, int_to_gpu(2)))
        result_add = gpu_to_int(calc.add(a, a))
        assert result_mul == result_add

    # -- Carry-stress: all-ones maximises FFT coefficient magnitude ---------

    @pytest.mark.parametrize("bits", [
        _FFT_ENTRY_BITS,
        _FFT_B16_BITS,
    ], ids=["entry", "b16"])
    def test_all_ones_carry_stress(self, calc, bits):
        """(2^n - 1)^2 produces maximum convolution coefficients."""
        a_int = (1 << bits) - 1
        result = gpu_to_int(calc.mul(int_to_gpu(a_int), int_to_gpu(a_int)))
        assert result == a_int * a_int

    # -- B=16 / B=8 chunk boundary -----------------------------------------

    @pytest.mark.parametrize("n", [
        1_900_000,  # well inside B=16 region
        2_100_000,  # just above estimated crossover (n_fft_est ≈ 2^22, B=8)
        4_000_000,  # clearly B=8
    ], ids=["1.9Mbit", "2.1Mbit", "4Mbit"])
    def test_chunk_boundary_power_of_2(self, calc, n):
        """Power-of-2 product straddling the B=16 / B=8 transition."""
        result = gpu_to_int(calc.mul(int_to_gpu(1 << n), int_to_gpu(1 << n)))
        assert result == (1 << (2 * n))


class TestMulFFTWorkspace:
    """Correctness of pre-allocated FFT workspace reuse (Phase 4)."""

    def test_buffer_grows_with_input_size(self):
        """Sequential multiplications with increasing size must all be correct."""
        calc = GPUBigInt()
        for bits in [_FFT_ENTRY_BITS, _FFT_B16_BITS, _FFT_B8_BITS]:
            n = bits // 2
            result = gpu_to_int(calc.mul(int_to_gpu(1 << n), int_to_gpu(1 << n)))
            assert result == (1 << (2 * n)), f"failed at {bits} bits"

    def test_buffer_reused_gives_same_result(self):
        """Two calls with identical inputs must return the same value (no stale data)."""
        calc = GPUBigInt()
        bits = _FFT_B16_BITS
        n = bits // 2
        a_gpu = int_to_gpu((1 << n) + 1)
        b_gpu = int_to_gpu((1 << n) - 1)
        r1 = gpu_to_int(calc.mul(a_gpu, b_gpu))
        r2 = gpu_to_int(calc.mul(a_gpu, b_gpu))
        expected = (1 << (2 * n)) - 1
        assert r1 == expected
        assert r2 == expected

    def test_descending_then_ascending_size(self):
        """Buffer should handle sizes that fluctuate (grow, shrink, grow)."""
        calc = GPUBigInt()
        sizes = [_FFT_B8_BITS, _FFT_ENTRY_BITS, _FFT_B16_BITS, _FFT_B8_BITS]
        for bits in sizes:
            n = bits // 2
            result = gpu_to_int(calc.mul(int_to_gpu(1 << n), int_to_gpu(1 << n)))
            assert result == (1 << (2 * n)), f"failed at {bits} bits"

    def test_workspace_shared_across_tabaiint_operations(self):
        """TabaiInt uses a module-level shared GPUBigInt; sequential muls must be correct."""
        bits = _FFT_B16_BITS
        n = bits // 2
        a = TabaiInt((1 << n) + 1)
        b = TabaiInt((1 << n) - 1)
        r1 = (a * b).to_cpu()
        r2 = (a * b).to_cpu()
        expected = (1 << (2 * n)) - 1
        assert r1 == expected
        assert r2 == expected


class TestMulFFTSquaring:
    """Phase 3: the FFT squaring fast path (mul(a, a) with the same array object
    skips the second operand's pad fill and forward transform, fa*=fa)."""

    @pytest.mark.parametrize("bits", [
        _FFT_ENTRY_BITS,   # B=16 FFT square (~10938 all-ones limbs)
        _FFT_B16_BITS,
        _FFT_B8_BITS,      # B=8 FFT square
    ], ids=["entry", "b16", "b8"])
    def test_square_all_ones(self, calc, bits):
        a_int = (1 << bits) - 1
        a = int_to_gpu(a_int)
        assert gpu_to_int(calc.mul(a, a)) == a_int * a_int

    def test_square_path_matches_general_path(self, calc):
        """mul(a, a) (is_square True) must equal mul(a, a_copy) (general path)."""
        random.seed(31337)
        for bits in [_FFT_ENTRY_BITS, _FFT_B16_BITS]:
            a_int = random.getrandbits(bits) | (1 << (bits - 1))
            a = int_to_gpu(a_int)
            a_copy = int_to_gpu(a_int)          # distinct object → general path
            assert a is not a_copy
            r_sq = gpu_to_int(calc.mul(a, a))    # a_gpu is b_gpu → square path
            r_gen = gpu_to_int(calc.mul(a, a_copy))
            assert r_sq == r_gen == a_int * a_int

    def test_square_asymmetric_result_length(self, calc):
        """Square of a value whose top limb is small (product shorter than 2*len)."""
        a_int = (1 << (_FFT_ENTRY_BITS - 30)) + 7   # top bits sparse
        a = int_to_gpu(a_int)
        assert gpu_to_int(calc.mul(a, a)) == a_int * a_int

    def test_tabaiint_pow2_uses_square_path(self):
        """TabaiInt x*x and x**2 at FFT sizes are correct (pow's squarings feed
        the same-object square path)."""
        n = _FFT_B16_BITS
        x = TabaiInt((1 << n) - 1)
        expected = ((1 << n) - 1) ** 2
        assert (x * x).to_cpu() == expected
        assert (x ** 2).to_cpu() == expected


# ---------------------------------------------------------------------------
# Edge case tests for GPUBigInt operations
# ---------------------------------------------------------------------------

class TestAddSubEdgeCases:
    """Edge cases for add/sub that aren't covered by boundary or exhaustive tests."""

    def test_add_zero_zero(self, calc):
        assert gpu_to_int(calc.add(int_to_gpu(0), int_to_gpu(0))) == 0

    def test_add_zero_left(self, calc):
        a = (1 << 1000) + 42
        assert gpu_to_int(calc.add(int_to_gpu(0), int_to_gpu(a))) == a

    def test_add_zero_right(self, calc):
        a = (1 << 1000) + 42
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(0))) == a

    def test_sub_to_zero(self, calc):
        a = (1 << 10000) - 1
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(a))) == 0

    def test_sub_zero_right(self, calc):
        a = (1 << 1000) + 42
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(0))) == a

    def test_add_asymmetric_sizes(self, calc):
        """One operand much larger than the other."""
        a = (1 << 50000) - 1  # ~1563 limbs
        b = 1                  # 1 limb
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b
        assert gpu_to_int(calc.add(int_to_gpu(b), int_to_gpu(a))) == a + b

    def test_sub_asymmetric_sizes(self, calc):
        a = (1 << 50000) - 1
        b = 1
        assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b

    def test_add_uint32_boundary_values(self, calc):
        """Values at uint32 limb boundaries."""
        vals = [2**32 - 1, 2**32, 2**32 + 1, 2**64 - 1, 2**64, 2**64 + 1]
        for a in vals:
            for b in vals:
                assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    def test_sub_uint32_boundary_values(self, calc):
        vals = [2**32 - 1, 2**32, 2**32 + 1, 2**64 - 1, 2**64, 2**64 + 1]
        for a in vals:
            for b in vals:
                if a >= b:
                    assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b

    def test_add_all_ones_different_lengths(self, calc):
        """0xFFFF...F + 0xFFFF...F where operands differ in limb count."""
        a = (1 << 320) - 1    # 10 limbs, all 0xFFFFFFFF
        b = (1 << 160) - 1    # 5 limbs, all 0xFFFFFFFF
        assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b

    def test_sub_leaves_single_bit(self, calc):
        """Result is exactly a power of 2."""
        a = (1 << 10000)
        b = a - 1
        result = gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b)))
        assert result == 1


class TestMulEdgeCases:
    """Edge cases for multiplication."""

    def test_mul_zero_zero(self, calc):
        assert gpu_to_int(calc.mul(int_to_gpu(0), int_to_gpu(0))) == 0

    def test_mul_one_one(self, calc):
        assert gpu_to_int(calc.mul(int_to_gpu(1), int_to_gpu(1))) == 1

    def test_mul_max_uint32(self, calc):
        a = 2**32 - 1
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(a))) == a * a

    def test_mul_power_of_2(self, calc):
        """Multiplying by a power of 2 is effectively a shift."""
        a = (1 << 5000) + 123456789
        b = 1 << 3000
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b

    def test_mul_asymmetric_sizes(self, calc):
        """One operand much larger than the other (low work → schoolbook path)."""
        a = (1 << 10000) - 1
        b = 3
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b

    def test_mul_asymmetric_small_times_large_schoolbook(self, calc):
        """Tiny x huge: work = 1*3125 << threshold, so schoolbook handles it
        even though one operand dwarfs the square dispatch boundary."""
        a = (1 << 100000) - 1   # ~3125 limbs
        b = 7                    # 1 limb
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b

    def test_mul_asymmetric_sizes_fft(self, calc):
        """Unbalanced but high-work pair (9375 x 3125 limbs, work ~29M >
        threshold) — forces the FFT path with asymmetric operand lengths."""
        a = (1 << 300000) - 1   # ~9375 limbs
        b = (1 << 100000) - 1   # ~3125 limbs
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b

    def test_mul_schoolbook_fft_threshold(self, calc):
        """Values straddling the schoolbook/FFT dispatch boundary (work=la*lb,
        isqrt(threshold)=5120)."""
        # 5120 limbs = 163840 bits, work = 5120^2 == threshold → schoolbook path
        a_sb = (1 << 163840) - 1
        b_sb = (1 << 163840) - 1
        assert gpu_to_int(calc.mul(int_to_gpu(a_sb), int_to_gpu(b_sb))) == a_sb * b_sb

        # 5121 limbs = 163872 bits, work = 5121^2 > threshold → FFT path
        a_fft = (1 << 163872) - 1
        b_fft = (1 << 163872) - 1
        assert gpu_to_int(calc.mul(int_to_gpu(a_fft), int_to_gpu(b_fft))) == a_fft * b_fft

    def test_mul_commutativity(self, calc):
        """a * b == b * a for various sizes."""
        random.seed(9999)
        for bits in [100, 5000, 80000]:
            a = random.getrandbits(bits) | (1 << (bits - 1))
            b = random.getrandbits(bits) | (1 << (bits - 1))
            r1 = gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b)))
            r2 = gpu_to_int(calc.mul(int_to_gpu(b), int_to_gpu(a)))
            assert r1 == r2

    def test_mul_single_limb_overflow(self, calc):
        """Products that overflow exactly at limb boundaries."""
        a = (1 << 32) - 1   # 0xFFFFFFFF
        b = (1 << 32) - 1
        expected = a * b     # 0xFFFFFFFE00000001
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == expected

    def test_mul_alternating_bits(self, calc):
        """Alternating bit pattern stresses the FFT differently from all-ones."""
        n = 80000  # FFT path
        a = sum(1 << i for i in range(0, n, 2))  # 0x5555...
        b = sum(1 << i for i in range(1, n, 2))  # 0xAAAA...
        assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b


class TestCompareEdgeCases:
    """Edge cases for GPUBigInt._compare."""

    def test_compare_equal(self, calc):
        a = (1 << 10000) - 1
        assert calc._compare(int_to_gpu(a), int_to_gpu(a)) == 0

    def test_compare_zero_zero(self, calc):
        assert calc._compare(int_to_gpu(0), int_to_gpu(0)) == 0

    def test_compare_differ_in_last_limb(self, calc):
        """Differ only in the most significant limb."""
        a = (1 << 10000) + (1 << 9999)
        b = (1 << 10000)
        assert calc._compare(int_to_gpu(a), int_to_gpu(b)) == 1
        assert calc._compare(int_to_gpu(b), int_to_gpu(a)) == -1

    def test_compare_differ_in_first_limb(self, calc):
        """Differ only in the least significant limb."""
        base = 1 << 10000
        a = base + 2
        b = base + 1
        assert calc._compare(int_to_gpu(a), int_to_gpu(b)) == 1
        assert calc._compare(int_to_gpu(b), int_to_gpu(a)) == -1

    def test_compare_different_lengths(self, calc):
        a = 1 << 10000   # many limbs
        b = 1             # 1 limb
        assert calc._compare(int_to_gpu(a), int_to_gpu(b)) == 1
        assert calc._compare(int_to_gpu(b), int_to_gpu(a)) == -1

    def test_compare_adjacent_values(self, calc):
        """Consecutive integers that differ by 1."""
        a = (1 << 5000) - 1
        b = 1 << 5000
        assert calc._compare(int_to_gpu(a), int_to_gpu(b)) == -1
        assert calc._compare(int_to_gpu(b), int_to_gpu(a)) == 1


class TestBitLengthEdgeCases:

    def test_bit_length_zero(self, calc):
        assert calc._bit_length(int_to_gpu(0)) == 0

    def test_bit_length_one(self, calc):
        assert calc._bit_length(int_to_gpu(1)) == 1

    def test_bit_length_power_of_2(self, calc):
        for exp in [1, 31, 32, 33, 63, 64, 65, 1000, 10000]:
            assert calc._bit_length(int_to_gpu(1 << exp)) == exp + 1

    def test_bit_length_power_of_2_minus_1(self, calc):
        for exp in [1, 31, 32, 33, 63, 64, 65, 1000, 10000]:
            assert calc._bit_length(int_to_gpu((1 << exp) - 1)) == exp


class TestShiftEdgeCases:

    def test_shift_left_zero_bits(self, calc):
        a = int_to_gpu(42)
        result = calc._shift_left(a, 0)
        assert gpu_to_int(result) == 42

    def test_shift_left_by_one(self, calc):
        a = (1 << 5000) + 1
        result = gpu_to_int(calc._shift_left(int_to_gpu(a), 1))
        assert result == a << 1

    def test_shift_left_by_32(self, calc):
        """Exact limb-aligned shift."""
        a = (1 << 5000) + 123
        result = gpu_to_int(calc._shift_left(int_to_gpu(a), 32))
        assert result == a << 32

    def test_shift_left_by_33(self, calc):
        """Non-aligned shift crossing limb boundary."""
        a = (1 << 5000) + 123
        result = gpu_to_int(calc._shift_left(int_to_gpu(a), 33))
        assert result == a << 33

    def test_shift_left_large(self, calc):
        a = (1 << 1000) - 1
        result = gpu_to_int(calc._shift_left(int_to_gpu(a), 5000))
        assert result == a << 5000

    def test_shift_right_one_basic(self, calc):
        a = 1024
        result = gpu_to_int(calc._shift_right_one(int_to_gpu(a)))
        assert result == 512

    def test_shift_right_one_odd(self, calc):
        """Shifting an odd number truncates the lowest bit."""
        a = (1 << 5000) + 1
        result = gpu_to_int(calc._shift_right_one(int_to_gpu(a)))
        assert result == a >> 1

    def test_shift_right_one_all_ones(self, calc):
        a = (1 << 10000) - 1
        result = gpu_to_int(calc._shift_right_one(int_to_gpu(a)))
        assert result == a >> 1

    def test_shift_right_one_power_of_2(self, calc):
        a = 1 << 10000
        result = gpu_to_int(calc._shift_right_one(int_to_gpu(a)))
        assert result == a >> 1

    def test_shift_right_one_one(self, calc):
        result = gpu_to_int(calc._shift_right_one(int_to_gpu(1)))
        assert result == 0


class TestDivmodEdgeCases:

    def test_divmod_equal(self, calc):
        a = (1 << 5000) - 1
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(a))
        assert gpu_to_int(q) == 1
        assert gpu_to_int(r) == 0

    def test_divmod_dividend_less_than_divisor(self, calc):
        a = 5
        b = (1 << 5000) - 1
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == 0
        assert gpu_to_int(r) == a

    def test_divmod_by_one(self, calc):
        a = (1 << 10000) + 42
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(1))
        assert gpu_to_int(q) == a
        assert gpu_to_int(r) == 0

    def test_divmod_by_zero_raises(self, calc):
        with pytest.raises(ZeroDivisionError):
            calc.divmod(int_to_gpu(42), int_to_gpu(0))

    def test_divmod_power_of_2_divisor(self, calc):
        a = (1 << 10000) + 999
        b = 1 << 100
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == a // b
        assert gpu_to_int(r) == a % b

    def test_divmod_large_quotient_small_remainder(self, calc):
        b = (1 << 1000) + 3
        a = b * 12345 + 7
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == 12345
        assert gpu_to_int(r) == 7

    def test_divmod_exact_division(self, calc):
        b = (1 << 2000) + 17
        a = b * 9999
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == 9999
        assert gpu_to_int(r) == 0

    def test_divmod_zero_dividend(self, calc):
        q, r = calc.divmod(int_to_gpu(0), int_to_gpu(42))
        assert gpu_to_int(q) == 0
        assert gpu_to_int(r) == 0


class TestDivmodNewtonPath:
    """divmod above _DIV_NEWTON_THRESHOLD_LIMBS (~164K bits) routes to the GPU
    Newton reciprocal.  These cases exercise that path against Python's exact
    divmod across a range of operand shapes that have historically trapped
    Newton-style algorithms."""

    # Sizes here must stay above _DIV_NEWTON_THRESHOLD_LIMBS so the GPU Newton
    # path is exercised, not the CPU shortcut.
    @pytest.mark.parametrize("a_bits,b_bits", [
        (500_000, 250_000),
        (500_000, 499_999),   # b nearly equal to a → small quotient
        (500_000, 10_000),    # b ≪ a → very large quotient
        (1_000_000, 500_000),
        (2_000_000, 1_000_000),
    ], ids=lambda v: f"{v}")
    def test_random_pair_matches_python(self, calc, a_bits, b_bits):
        random.seed(a_bits * 7919 + b_bits)
        a = (1 << (a_bits - 1)) | _rand_bits(a_bits - 1)
        b = (1 << (b_bits - 1)) | _rand_bits(b_bits - 1)
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        eq, er = divmod(a, b)
        assert gpu_to_int(q) == eq
        assert gpu_to_int(r) == er

    def test_b_is_power_of_2(self, calc):
        # Power-of-2 b is a worst case for the float64 seed: only one bit set,
        # so b_top + 1 collapses to the next power of 2.
        a = _rand_bits(800_000) | (1 << 799_999)
        b = 1 << 400_000
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == a // b
        assert gpu_to_int(r) == a % b

    def test_b_all_ones(self, calc):
        # b = 2^k - 1 maximizes b's top 53 bits, stressing the seed scaling.
        a = _rand_bits(800_000) | (1 << 799_999)
        b = (1 << 400_000) - 1
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == a // b
        assert gpu_to_int(r) == a % b

    def test_a_just_above_b(self, calc):
        # Quotient is exactly 1 with a small remainder — the post-Newton
        # one-sided correction path needs to handle this without underflow.
        b = _rand_bits(500_000) | (1 << 499_999)
        a = b + 1
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == 1
        assert gpu_to_int(r) == 1

    def test_exact_multiple(self, calc):
        # r should be exactly 0 — verifies no spurious +1 correction.
        b = _rand_bits(500_000) | (1 << 499_999)
        k = _rand_bits(300_000) | (1 << 299_999)
        a = b * k
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == k
        assert gpu_to_int(r) == 0


class TestNewtonRampTrimGate:
    """The Newton ramp normalises each step's width one of two ways depending on
    _NEWTON_RAMP_TRIM_LIMBS: a deterministic sync-free top slice (small x) or a
    _trim (large x).  Force each branch via the gate and require both to agree
    with Python's exact divmod — including the power-of-two divisor edge, where
    b_trunc is a bare power of two and the deterministic slice must keep the full
    q_field//32 + 2 limbs (a tighter width would drop the top bit)."""

    # gate=0 forces every ramp step through _trim; a huge gate forces every step
    # through the deterministic slice.  Both must be exact.
    @pytest.mark.parametrize("gate", [0, 10**9])
    @pytest.mark.parametrize("a_bits,b_bits", [
        (400_000, 200_000),   # generic Newton-size divmod
        (400_000, 200_001),   # b_bits % 32 == 1  → q_field % 32 == 31 alignment
    ], ids=lambda v: f"{v}")
    def test_both_branches_match_python(self, calc, monkeypatch, gate, a_bits, b_bits):
        monkeypatch.setattr("tabai_gpu.core._NEWTON_RAMP_TRIM_LIMBS", gate)
        random.seed(a_bits * 31 + b_bits + gate)
        a = _rand_bits(a_bits) | (1 << (a_bits - 1))
        b = _rand_bits(b_bits) | (1 << (b_bits - 1))
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == a // b
        assert gpu_to_int(r) == a % b

    @pytest.mark.parametrize("gate", [0, 10**9])
    def test_power_of_two_divisor_both_branches(self, calc, monkeypatch, gate):
        monkeypatch.setattr("tabai_gpu.core._NEWTON_RAMP_TRIM_LIMBS", gate)
        a = _rand_bits(400_000) | (1 << 399_999)
        b = 1 << 200_001          # power of two, b.bit_length() % 32 == 2
        q, r = calc.divmod(int_to_gpu(a), int_to_gpu(b))
        assert gpu_to_int(q) == a // b
        assert gpu_to_int(r) == a % b

    def test_gate_choice_does_not_change_result(self, calc, monkeypatch):
        """The deterministic slice and _trim must produce identical quotients."""
        random.seed(20260706)
        a = _rand_bits(600_000) | (1 << 599_999)
        b = _rand_bits(300_000) | (1 << 299_999)
        ga = int_to_gpu(a)
        gb = int_to_gpu(b)
        monkeypatch.setattr("tabai_gpu.core._NEWTON_RAMP_TRIM_LIMBS", 0)
        q_trim, r_trim = calc.divmod(ga, gb)
        monkeypatch.setattr("tabai_gpu.core._NEWTON_RAMP_TRIM_LIMBS", 10**9)
        q_det, r_det = calc.divmod(ga, gb)
        assert gpu_to_int(q_trim) == gpu_to_int(q_det) == a // b
        assert gpu_to_int(r_trim) == gpu_to_int(r_det) == a % b


def _rand_bits(n: int) -> int:
    """random.getrandbits substitute that works for n > 2**31."""
    if n <= 0:
        return 0
    chunk = 1 << 29
    out = 0
    remaining = n
    while remaining > 0:
        k = min(remaining, chunk)
        out = (out << k) | random.getrandbits(k)
        remaining -= k
    return out


class TestUtilsRoundtrip:
    """int_to_gpu / gpu_to_int roundtrip edge cases."""

    @pytest.mark.parametrize("val", [
        0, 1, 2,
        2**16 - 1, 2**16, 2**16 + 1,
        2**31 - 1, 2**31, 2**31 + 1,
        2**32 - 1, 2**32, 2**32 + 1,
        2**64 - 1, 2**64, 2**64 + 1,
        2**128 - 1,
        (1 << 10000) - 1, 1 << 10000,
    ])
    def test_roundtrip(self, val):
        assert gpu_to_int(int_to_gpu(val)) == val

    def test_roundtrip_random(self):
        random.seed(54321)
        for bits in [1, 8, 16, 31, 32, 33, 64, 128, 1000, 10000]:
            val = random.getrandbits(bits)
            assert gpu_to_int(int_to_gpu(val)) == val
