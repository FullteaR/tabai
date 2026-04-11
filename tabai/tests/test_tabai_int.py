import pytest
import random
from tabai_gpu import TabaiInt


def test_add_1_1():
    assert (TabaiInt(1) + TabaiInt(1)).to_cpu() == 2


def test_add_basic():
    a, b = 10**100, 20**100
    assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b


def test_add_small_numbers():
    for a in range(0, 100):
        for b in range(0, 100):
            assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b


def test_add_carry_chain():
    a = (1 << 100000) - 1
    assert (TabaiInt(a) + TabaiInt(1)).to_cpu() == 1 << 100000


def test_sub_basic():
    a, b = 20**100, 10**100
    assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b


def test_mul_1_1():
    assert (TabaiInt(1) * TabaiInt(1)).to_cpu() == 1


def test_mul_basic():
    a, b = 10**100, 20**100
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b


def test_mul_by_zero():
    assert (TabaiInt(10**100) * TabaiInt(0)).to_cpu() == 0


def test_mul_by_one():
    a = 10**100
    assert (TabaiInt(a) * TabaiInt(1)).to_cpu() == a


def test_mul_small_numbers():
    for a in range(0, 100):
        for b in range(0, 100):
            assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b


def test_mul_carry_chain():
    a = (1 << 100000) - 1
    b = (1 << 100000) - 1
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b


def test_repr():
    assert repr(TabaiInt(42)) == "TabaiInt(42)"


def test_from_int_and_to_cpu_roundtrip():
    values = [0, 1, 2**32 - 1, 2**32, 2**64, 10**100]
    for v in values:
        assert TabaiInt(v).to_cpu() == v


def test_floordiv_basic():
    a, b = 7, 3
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == 2


def test_mod_basic():
    a, b = 7, 3
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == 1


def test_divmod_basic():
    a, b = 7, 3
    q, r = divmod(TabaiInt(a), TabaiInt(b))
    assert q.to_cpu() == 2
    assert r.to_cpu() == 1


def test_div_exact():
    a, b = 6, 3
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == 2
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == 0


def test_div_small_numbers():
    for a in range(0, 100):
        for b in range(1, 100):
            assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == a // b
            assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == a % b


def test_div_zero_dividend():
    assert (TabaiInt(0) // TabaiInt(5)).to_cpu() == 0
    assert (TabaiInt(0) % TabaiInt(5)).to_cpu() == 0


def test_div_by_one():
    a = 10**100
    assert (TabaiInt(a) // TabaiInt(1)).to_cpu() == a
    assert (TabaiInt(a) % TabaiInt(1)).to_cpu() == 0


def test_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        TabaiInt(7) // TabaiInt(0)


def test_div_large():
    a, b = 20**100, 10**100
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == a // b
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == a % b


@pytest.mark.parametrize("a,b", [
    ((1 << 1000) - (1 << 500) + 1, (1 << 999) + (1 << 333) - 1),
    ((1 << 100000) - (1 << 50000) + 1, (1 << 99999) + (1 << 33333) - 1),
], ids=["1000bit", "100000bit"])
def test_large_ops(a, b):
    assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b
    assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == a // b
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == a % b


def test_pow_basic():
    assert (TabaiInt(2) ** TabaiInt(5)).to_cpu() == 32
    assert (TabaiInt(2) ** 4).to_cpu() == 16
    assert (3 ** TabaiInt(5)).to_cpu() == 243


def test_pow_zero_exponent():
    assert (TabaiInt(5) ** 0).to_cpu() == 1
    assert (TabaiInt(0) ** 0).to_cpu() == 1


def test_pow_one_exponent():
    assert (TabaiInt(42) ** 1).to_cpu() == 42


def test_pow_zero_base():
    assert (TabaiInt(0) ** 5).to_cpu() == 0


def test_pow_one_base():
    assert (TabaiInt(1) ** 100).to_cpu() == 1


def test_pow_negative_exponent():
    with pytest.raises(ValueError):
        TabaiInt(2) ** TabaiInt(-3)


def test_pow_small_numbers():
    for a in range(0, 20):
        for b in range(0, 10):
            assert (TabaiInt(a) ** TabaiInt(b)).to_cpu() == a ** b


def test_pow_large():
    a, b = 10**100, 3
    assert (TabaiInt(a) ** TabaiInt(b)).to_cpu() == a ** b


# ---------------------------------------------------------------------------
# Boundary tests for the fused add/sub kernel (256-limb threshold)
# ---------------------------------------------------------------------------

_SMALL_BITS = 8192   # 256 limbs — fused-kernel path
_LARGE_BITS = 8193   # 257 limbs — multi-kernel path


class TestAddSubBoundary:
    """Correctness around the 256-limb boundary, exercised through TabaiInt."""

    def test_add_at_limit(self):
        a = (1 << (_SMALL_BITS - 1))
        b = (1 << (_SMALL_BITS - 2))
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_add_carry_chain_fills_256_limbs(self):
        a = (1 << _SMALL_BITS) - 1
        assert (TabaiInt(a) + TabaiInt(1)).to_cpu() == 1 << _SMALL_BITS

    def test_add_carry_out_creates_257th_limb(self):
        a = (1 << _SMALL_BITS) - 1
        b = (1 << _SMALL_BITS) - 1
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_sub_at_limit(self):
        a = (1 << _SMALL_BITS) - 1
        b = (1 << (_SMALL_BITS // 2))
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b

    def test_sub_borrow_chain_fills_256_limbs(self):
        a = 1 << _SMALL_BITS
        assert (TabaiInt(a) - TabaiInt(1)).to_cpu() == a - 1

    def test_add_just_above_limit(self):
        a = 1 << _LARGE_BITS
        assert (TabaiInt(a) + TabaiInt(1)).to_cpu() == a + 1

    def test_add_carry_chain_257_limbs(self):
        a = (1 << _LARGE_BITS) - 1
        assert (TabaiInt(a) + TabaiInt(1)).to_cpu() == 1 << _LARGE_BITS

    def test_sub_just_above_limit(self):
        a = (1 << _LARGE_BITS) - 1
        b = (1 << (_LARGE_BITS // 2))
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b

    def test_add_negative_at_limit(self):
        a = -((1 << _SMALL_BITS) - 1)
        b = -1
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_sub_negative_at_limit(self):
        a = (1 << _SMALL_BITS) - 1
        b = -((1 << _SMALL_BITS) - 1)
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b

    @pytest.mark.parametrize("bits", [
        _SMALL_BITS - 32,
        _SMALL_BITS - 1,
        _SMALL_BITS,
        _SMALL_BITS + 1,
        _SMALL_BITS + 32,
    ])
    def test_add_random_around_boundary(self, bits):
        random.seed(bits)
        a = random.getrandbits(bits) | (1 << (bits - 1))
        b = random.getrandbits(bits) | (1 << (bits - 1))
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    @pytest.mark.parametrize("bits", [
        _SMALL_BITS - 32,
        _SMALL_BITS - 1,
        _SMALL_BITS,
        _SMALL_BITS + 1,
        _SMALL_BITS + 32,
    ])
    def test_sub_random_around_boundary(self, bits):
        random.seed(bits)
        a = random.getrandbits(bits) | (1 << bits)
        b = random.getrandbits(bits) | (1 << (bits - 1))
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b


# ---------------------------------------------------------------------------
# Large-precision multiplication tests (FFT chunk-width adaptive path)
#
# These tests verify correctness of the B=16 → B=8 adaptive mul without
# requiring Python to multiply large integers (which would be extremely slow).
# Instead they rely on identities whose expected values can be formed with
# cheap Python bit-shifts:
#
#   Identity 1  — power-of-2 product:
#       2^n  *  2^m  =  2^(n+m)
#
#   Identity 2  — difference-of-squares:
#       (2^n + 1) * (2^n - 1)  =  2^(2n) - 1
#
#   Identity 3  — self-consistency with add:
#       a * 2  ==  a + a
#
# Bit-size coverage:
#   _MUL_B16_BITS  — inside the B=16 path (n_fft_est < 2^20, ≲ 4M-bit operands)
#   _MUL_B8_BITS   — inside the B= 8 path (n_fft_est ≥ 2^20, 10M-bit operands)
#   _MUL_LARGE_BITS — even larger (100M-bit operands, stresses GPU memory)
# ---------------------------------------------------------------------------

_MUL_B16_BITS  = 2_000_000   # 2M bits — B=16 path
_MUL_B8_BITS   = 10_000_000  # 10M bits — B=8 path
_MUL_LARGE_BITS = 100_000_000 # 100M bits — B=8, tests GPU memory at scale


class TestMulLargePrecision:
    """Verify FFT-based multiplication at sizes where the B=16 path loses float64 precision."""

    # ---- Identity 1: 2^n * 2^m = 2^(n+m) --------------------------------

    @pytest.mark.parametrize("n,m", [
        (_MUL_B16_BITS // 2, _MUL_B16_BITS // 2),
        (_MUL_B8_BITS  // 2, _MUL_B8_BITS  // 2),
    ], ids=["2Mbit", "10Mbit"])
    def test_power_of_2_product(self, n, m):
        result = (TabaiInt(1 << n) * TabaiInt(1 << m)).to_cpu()
        assert result == (1 << (n + m))

    # ---- Identity 2: (2^n + 1)(2^n - 1) = 2^(2n) - 1 -------------------

    @pytest.mark.parametrize("n", [
        _MUL_B16_BITS // 2,
        _MUL_B8_BITS  // 2,
    ], ids=["2Mbit", "10Mbit"])
    def test_diff_of_squares(self, n):
        a = TabaiInt((1 << n) + 1)
        b = TabaiInt((1 << n) - 1)
        result = (a * b).to_cpu()
        assert result == (1 << (2 * n)) - 1

    # ---- Identity 3: a * 2 == a + a (self-consistency) ------------------

    @pytest.mark.parametrize("bits", [
        _MUL_B16_BITS,
        _MUL_B8_BITS,
        _MUL_LARGE_BITS,
    ], ids=["2Mbit", "10Mbit", "100Mbit"])
    def test_mul2_equals_add(self, bits):
        random.seed(bits)
        a_int = random.getrandbits(bits) | (1 << (bits - 1))
        a = TabaiInt(a_int)
        assert (a * TabaiInt(2)).to_cpu() == (a + a).to_cpu()

    # ---- Chunk-width boundary: same result just below and just above -----

    @pytest.mark.parametrize("n", [
        1_900_000,  # well inside B=16 region
        2_100_000,  # just above the estimated crossover
        4_000_000,  # clearly in B=8 region
    ], ids=["1.9Mbit", "2.1Mbit", "4Mbit"])
    def test_chunk_boundary_power_of_2(self, n):
        """Power-of-2 product straddling the B=16 / B=8 transition."""
        result = (TabaiInt(1 << n) * TabaiInt(1 << n)).to_cpu()
        assert result == (1 << (2 * n))
