"""Boundary-condition unit tests for TabaiInt.

Covers limb (uint32) boundaries, kernel-path thresholds, and
operator edge cases that straddle architectural switch points.
Does NOT modify src.
"""

import pytest
import random
from tabai_gpu import TabaiInt

import sys
sys.set_int_max_str_digits(0)

# ---------------------------------------------------------------------------
# Limb-boundary constants
# ---------------------------------------------------------------------------
_U32_MAX = 2**32 - 1
_U32 = 2**32
_U64_MAX = 2**64 - 1
_U64 = 2**64

# CPU/FFT mul threshold  (core.py: _MUL_CPU_THRESHOLD_LIMBS = 2048)
_MUL_CPU_BITS = 2048 * 32        # 65536 bits — last CPU-path size
_MUL_FFT_BITS = 2049 * 32        # 65568 bits — first FFT-path size

# Newton reciprocal threshold for divmod  (core.py: _DIV_NEWTON_THRESHOLD_LIMBS = 12288)
_NEWTON_LIMBS = 12288
_NEWTON_BITS = _NEWTON_LIMBS * 32  # 393216 bits

# Pow sliding-window k thresholds (based on exponent bit_length)
#   k=1 for bit_length <= 8,  k=2 for <= 24,  k=3 for <= 70, ...
_POW_K1_MAX_EXP = (1 << 8) - 1    # 255
_POW_K2_MIN_EXP = 1 << 8          # 256
_POW_K2_MAX_EXP = (1 << 24) - 1   # 16_777_215


# =========================================================================
# 1. Construction roundtrip at limb boundaries
# =========================================================================

class TestLimbBoundaryConstruction:

    @pytest.mark.parametrize("val", [
        0, 1, -1,
        _U32_MAX, -_U32_MAX,
        _U32, -_U32,
        _U32 + 1, -(_U32 + 1),
        _U64_MAX, -_U64_MAX,
        _U64, -_U64,
        _U64 + 1, -(_U64 + 1),
        (1 << 96) - 1, -((1 << 96) - 1),
        1 << 96, -(1 << 96),
        (1 << 128) - 1, -((1 << 128) - 1),
        1 << 128, -(1 << 128),
    ])
    def test_roundtrip(self, val):
        assert TabaiInt(val).to_cpu() == val


# =========================================================================
# 2. Add / Sub at limb boundaries (carry / borrow propagation)
# =========================================================================

class TestAddLimbBoundary:

    @pytest.mark.parametrize("a,b", [
        (_U32_MAX, 1),                            # carry into 2nd limb
        (_U64_MAX, 1),                            # carry into 3rd limb
        ((1 << 96) - 1, 1),                       # carry into 4th limb
        (_U32_MAX, _U32_MAX),                     # 1-limb + 1-limb → 2-limb
        (_U32, _U32),                             # 2-limb + 2-limb
        (_U64_MAX, _U64_MAX),                     # 2-limb + 2-limb → 3-limb
    ])
    def test_add_carry_at_limb_boundary(self, a, b):
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    @pytest.mark.parametrize("a,b", [
        (_U32_MAX, -1),
        (_U32, -1),
        (_U64_MAX, -1),
        (_U64, -1),
        (-_U32_MAX, 1),
        (-_U32, 1),
        (-_U32_MAX, -_U32_MAX),
        (-_U64_MAX, -1),
    ])
    def test_add_signed_at_limb_boundary(self, a, b):
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b


class TestSubLimbBoundary:

    @pytest.mark.parametrize("a,b", [
        (_U32, 1),                                # borrow from 2nd limb
        (_U64, 1),                                # borrow from 3rd limb
        (1 << 96, 1),                             # borrow from 4th limb
        (_U32, _U32_MAX),                         # 2-limb - 1-limb = 1
        (_U64, _U64_MAX),                         # 3-limb - 2-limb = 1
    ])
    def test_sub_borrow_at_limb_boundary(self, a, b):
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b

    @pytest.mark.parametrize("bits", [32, 64, 96, 128])
    def test_sub_to_zero_at_boundary(self, bits):
        v = (1 << bits) - 1
        assert (TabaiInt(v) - TabaiInt(v)).to_cpu() == 0

    @pytest.mark.parametrize("a,b", [
        (-_U32, -1),                              # (-2^32) - (-1)  = -2^32 + 1
        (1, _U32),                                # 1 - 2^32  = -(2^32 - 1)
        (-1, -_U32_MAX),                          # -1 - (-(2^32-1)) = 2^32-2
    ])
    def test_sub_signed_at_limb_boundary(self, a, b):
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b


# =========================================================================
# 3. Mul at limb boundaries
# =========================================================================

class TestMulLimbBoundary:

    @pytest.mark.parametrize("a,b", [
        (_U32_MAX, _U32_MAX),                     # (2^32-1)^2
        (_U32_MAX, 2),                            # doubles across limb
        (_U32_MAX, _U32),                         # 1-limb × 2-limb
        (_U32, _U32),                             # 2^64
        (_U64_MAX, _U64_MAX),                     # (2^64-1)^2
        (_U64_MAX, 2),
        (1, _U64_MAX),
    ])
    def test_mul_at_limb_boundary(self, a, b):
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_mul_negative_at_limb_boundary(self):
        assert (TabaiInt(-_U32_MAX) * TabaiInt(_U32_MAX)).to_cpu() == -(_U32_MAX ** 2)
        assert (TabaiInt(-_U32_MAX) * TabaiInt(-_U32_MAX)).to_cpu() == _U32_MAX ** 2
        assert (TabaiInt(-_U32) * TabaiInt(2)).to_cpu() == -2 * _U32


# =========================================================================
# 4. Mul at CPU / FFT path threshold (2048 limbs)
# =========================================================================

class TestMulCpuFftThreshold:

    def test_mul_at_cpu_threshold(self):
        a = (1 << _MUL_CPU_BITS) - 1
        b = (1 << _MUL_CPU_BITS) - 1
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_mul_just_above_cpu_threshold(self):
        a = (1 << _MUL_FFT_BITS) - 1
        b = (1 << _MUL_FFT_BITS) - 1
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_mul_mixed_cpu_fft_operand(self):
        a = (1 << _MUL_CPU_BITS) - 1    # 2048 limbs → CPU eligible
        b = (1 << _MUL_FFT_BITS) - 1    # 2049 limbs → forces FFT
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_mul_identity_at_cpu_threshold(self):
        n = _MUL_CPU_BITS // 2
        a = (1 << n) + 1
        b = (1 << n) - 1
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == (1 << (2 * n)) - 1


# =========================================================================
# 5. Divmod at limb boundaries
# =========================================================================

class TestDivmodLimbBoundary:

    @pytest.mark.parametrize("a,b", [
        (_U32, 1),
        (_U32, _U32_MAX),                        # quotient=1, remainder=1
        (_U64, _U32),
        (_U64_MAX, _U32_MAX),
        (_U64, _U64_MAX),                        # quotient=1, remainder=1
    ])
    def test_divmod_at_limb_boundary(self, a, b):
        q, r = divmod(TabaiInt(a), TabaiInt(b))
        eq, er = divmod(a, b)
        assert q.to_cpu() == eq
        assert r.to_cpu() == er

    @pytest.mark.parametrize("a,b", [
        (-_U32, 1),
        (-_U32, _U32_MAX),
        (_U32, -_U32_MAX),
        (-_U64, _U32),
        (-_U32_MAX, -1),
    ])
    def test_divmod_signed_at_limb_boundary(self, a, b):
        q, r = divmod(TabaiInt(a), TabaiInt(b))
        eq, er = divmod(a, b)
        assert q.to_cpu() == eq
        assert r.to_cpu() == er

    @pytest.mark.parametrize("val", [_U32_MAX, _U32, _U64_MAX, _U64])
    def test_divmod_self_at_boundary(self, val):
        q, r = divmod(TabaiInt(val), TabaiInt(val))
        assert q.to_cpu() == 1
        assert r.to_cpu() == 0

    @pytest.mark.parametrize("val", [_U32, _U64, 1 << 128])
    def test_divmod_dividend_one_less_than_divisor(self, val):
        q, r = divmod(TabaiInt(val - 1), TabaiInt(val))
        assert q.to_cpu() == 0
        assert r.to_cpu() == val - 1

    def test_divmod_identity_at_limb_boundary(self):
        pairs = [
            (_U32_MAX, 1),
            (_U32, _U32_MAX),
            (_U64_MAX, _U32_MAX),
            (-_U32_MAX, _U32),
            (_U32_MAX, -_U32),
        ]
        for a, b in pairs:
            ta, tb = TabaiInt(a), TabaiInt(b)
            q = ta // tb
            r = ta % tb
            assert (q * tb + r).to_cpu() == a


# =========================================================================
# 6. Divmod at Newton reciprocal threshold
# =========================================================================

class TestDivmodNewtonThreshold:

    def test_divmod_at_cpu_reciprocal(self):
        """a_bits = _NEWTON_BITS → p_limbs = _NEWTON_LIMBS → CPU reciprocal."""
        random.seed(0xDEAD)
        a_bits = _NEWTON_BITS
        b_bits = a_bits // 2
        a = random.getrandbits(a_bits) | (1 << (a_bits - 1))
        b = random.getrandbits(b_bits) | (1 << (b_bits - 1))
        q, r = divmod(TabaiInt(a), TabaiInt(b))
        eq, er = divmod(a, b)
        assert q.to_cpu() == eq
        assert r.to_cpu() == er

    def test_divmod_at_newton_reciprocal(self):
        """a_bits = _NEWTON_BITS + 32 → p_limbs > _NEWTON_LIMBS → GPU Newton."""
        random.seed(0xBEEF)
        a_bits = _NEWTON_BITS + 32
        b_bits = a_bits // 2
        a = random.getrandbits(a_bits) | (1 << (a_bits - 1))
        b = random.getrandbits(b_bits) | (1 << (b_bits - 1))
        q, r = divmod(TabaiInt(a), TabaiInt(b))
        eq, er = divmod(a, b)
        assert q.to_cpu() == eq
        assert r.to_cpu() == er

    def test_divmod_identity_across_newton_boundary(self):
        """a == q*b + r must hold on both sides of the Newton threshold."""
        random.seed(42)
        for a_bits in [_NEWTON_BITS, _NEWTON_BITS + 32]:
            b_bits = a_bits // 2
            a = random.getrandbits(a_bits) | (1 << (a_bits - 1))
            b = random.getrandbits(b_bits) | (1 << (b_bits - 1))
            ta, tb = TabaiInt(a), TabaiInt(b)
            q = ta // tb
            r = ta % tb
            assert (q * tb + r).to_cpu() == a


# =========================================================================
# 7. Pow sliding-window k boundary
# =========================================================================

class TestPowWindowBoundary:
    """Test exponents right at the k=1/k=2 and k=2/k=3 transitions."""

    def test_pow_k1_max(self):
        """bit_length(255) = 8 → k=1."""
        assert (TabaiInt(3) ** TabaiInt(_POW_K1_MAX_EXP)).to_cpu() == 3 ** _POW_K1_MAX_EXP

    def test_pow_k2_min(self):
        """bit_length(256) = 9 → k=2."""
        assert (TabaiInt(3) ** TabaiInt(_POW_K2_MIN_EXP)).to_cpu() == 3 ** _POW_K2_MIN_EXP

    def test_pow_k2_to_k3_boundary(self):
        """bit_length(2^24 - 1) = 24 → k=2;  bit_length(2^24) = 25 → k=3.
        Verify via 2^exp (cheap to check with bit-shift)."""
        for exp in [_POW_K2_MAX_EXP, _POW_K2_MAX_EXP + 1]:
            assert (TabaiInt(2) ** TabaiInt(exp)).to_cpu() == 1 << exp

    def test_pow_negative_base_at_window_boundary(self):
        assert (TabaiInt(-3) ** TabaiInt(_POW_K1_MAX_EXP)).to_cpu() == (-3) ** _POW_K1_MAX_EXP
        assert (TabaiInt(-3) ** TabaiInt(_POW_K2_MIN_EXP)).to_cpu() == (-3) ** _POW_K2_MIN_EXP

    @pytest.mark.parametrize("base", [0, 1, -1, _U32_MAX, -_U32_MAX, _U32])
    def test_pow_zero_exponent(self, base):
        assert (TabaiInt(base) ** TabaiInt(0)).to_cpu() == 1

    @pytest.mark.parametrize("base", [0, 1, -1, _U32_MAX, -_U32_MAX, _U32, -_U32])
    def test_pow_one_exponent(self, base):
        assert (TabaiInt(base) ** TabaiInt(1)).to_cpu() == base

    def test_pow_base_at_limb_boundary(self):
        assert (TabaiInt(_U32_MAX) ** TabaiInt(2)).to_cpu() == _U32_MAX ** 2
        assert (TabaiInt(_U32) ** TabaiInt(2)).to_cpu() == _U32 ** 2
        assert (TabaiInt(-_U32_MAX) ** TabaiInt(3)).to_cpu() == (-_U32_MAX) ** 3


# =========================================================================
# 8. Comparison at limb boundaries
# =========================================================================

class TestCompareLimbBoundary:

    def test_compare_across_limb_count_change(self):
        assert TabaiInt(_U32_MAX) < TabaiInt(_U32)
        assert TabaiInt(_U32) > TabaiInt(_U32_MAX)
        assert TabaiInt(_U64_MAX) < TabaiInt(_U64)

    @pytest.mark.parametrize("boundary", [_U32, _U64, 1 << 96, 1 << 128])
    def test_compare_adjacent_at_boundary(self, boundary):
        a = TabaiInt(boundary - 1)
        b = TabaiInt(boundary)
        c = TabaiInt(boundary + 1)
        assert a < b < c
        assert c > b > a
        assert a != b and b != c

    def test_compare_negative_at_limb_boundary(self):
        assert TabaiInt(-_U32_MAX) < TabaiInt(-1)
        assert TabaiInt(-_U32) < TabaiInt(-_U32_MAX)
        assert TabaiInt(-_U64) < TabaiInt(-_U32)

    @pytest.mark.parametrize("val", [_U32_MAX, _U32, _U64_MAX, _U64])
    def test_eq_and_ne_at_boundary(self, val):
        assert TabaiInt(val) == TabaiInt(val)
        assert TabaiInt(val) != TabaiInt(val + 1)
        assert TabaiInt(val) != TabaiInt(val - 1)
        assert TabaiInt(val) != TabaiInt(-val)


# =========================================================================
# 9. Reverse (int op TabaiInt) at limb boundaries
# =========================================================================

class TestReverseOpsLimbBoundary:

    @pytest.mark.parametrize("a,b", [
        (1, _U32_MAX),
        (_U32_MAX, 1),
        (0, _U32),
        (_U32 + 1, _U32),
    ])
    def test_radd(self, a, b):
        assert (a + TabaiInt(b)).to_cpu() == a + b

    @pytest.mark.parametrize("a,b", [
        (_U32, 1),
        (0, _U32_MAX),
        (_U64, _U32),
    ])
    def test_rsub(self, a, b):
        assert (a - TabaiInt(b)).to_cpu() == a - b

    @pytest.mark.parametrize("a,b", [
        (2, _U32_MAX),
        (_U32_MAX, 2),
        (0, _U32),
    ])
    def test_rmul(self, a, b):
        assert (a * TabaiInt(b)).to_cpu() == a * b

    @pytest.mark.parametrize("a,b", [
        (_U32, _U32_MAX),
        (_U64, _U32),
        (0, 1),
    ])
    def test_rfloordiv(self, a, b):
        assert (a // TabaiInt(b)).to_cpu() == a // b

    @pytest.mark.parametrize("a,b", [
        (_U32, _U32_MAX),
        (_U64, _U32),
    ])
    def test_rmod(self, a, b):
        assert (a % TabaiInt(b)).to_cpu() == a % b

    @pytest.mark.parametrize("a,b", [
        (2, 1000),
        (_U32_MAX, 2),
        (3, 255),
    ])
    def test_rpow(self, a, b):
        assert (a ** TabaiInt(b)).to_cpu() == a ** b


# =========================================================================
# 10. Sign boundary — neg / abs / zero normalisation
# =========================================================================

class TestSignBoundary:

    def test_neg_at_limb_boundary(self):
        for val in [_U32_MAX, _U32, _U64_MAX, _U64]:
            assert (-TabaiInt(val)).to_cpu() == -val
            assert (-TabaiInt(-val)).to_cpu() == val

    def test_abs_at_limb_boundary(self):
        for val in [_U32_MAX, _U32, _U64_MAX, _U64]:
            assert abs(TabaiInt(-val)).to_cpu() == val
            assert abs(TabaiInt(val)).to_cpu() == val

    def test_zero_sign_after_operations(self):
        for val in [_U32_MAX, _U32, _U64_MAX]:
            r = TabaiInt(val) - TabaiInt(val)
            assert r.to_cpu() == 0
            assert r._sign == 1

            r = TabaiInt(-val) + TabaiInt(val)
            assert r.to_cpu() == 0
            assert r._sign == 1

            r = TabaiInt(0) * TabaiInt(val)
            assert r.to_cpu() == 0

    def test_neg_zero_is_zero(self):
        z = -TabaiInt(0)
        assert z.to_cpu() == 0
        assert z._sign == 1


# =========================================================================
# 11. Unsupported-type coercion boundary
# =========================================================================

class TestCoercionBoundary:

    def test_add_unsupported_type(self):
        assert TabaiInt(1).__add__("x") is NotImplemented

    def test_sub_unsupported_type(self):
        assert TabaiInt(1).__sub__("x") is NotImplemented

    def test_mul_unsupported_type(self):
        assert TabaiInt(1).__mul__("x") is NotImplemented

    def test_floordiv_unsupported_type(self):
        assert TabaiInt(1).__floordiv__("x") is NotImplemented

    def test_mod_unsupported_type(self):
        assert TabaiInt(1).__mod__("x") is NotImplemented

    def test_pow_unsupported_type(self):
        assert TabaiInt(1).__pow__("x") is NotImplemented

    def test_eq_unsupported_type(self):
        assert (TabaiInt(1) == "x") is False
        assert (TabaiInt(1) != "x") is True


# =========================================================================
# 12. Floordiv / mod consistency at limb boundaries
# =========================================================================

class TestFloordivModConsistency:
    """floordiv and mod should individually agree with divmod."""

    @pytest.mark.parametrize("a,b", [
        (_U32_MAX, 3),
        (_U32, 3),
        (_U64_MAX, _U32_MAX),
        (-_U32_MAX, 3),
        (_U32_MAX, -3),
        (-_U32_MAX, -3),
    ])
    def test_floordiv_mod_match_divmod(self, a, b):
        ta, tb = TabaiInt(a), TabaiInt(b)
        q_dm, r_dm = divmod(ta, tb)
        assert (ta // tb).to_cpu() == q_dm.to_cpu()
        assert (ta % tb).to_cpu() == r_dm.to_cpu()

    def test_divmod_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            TabaiInt(_U32_MAX) // TabaiInt(0)
        with pytest.raises(ZeroDivisionError):
            TabaiInt(_U32_MAX) % TabaiInt(0)
        with pytest.raises(ZeroDivisionError):
            divmod(TabaiInt(_U32_MAX), TabaiInt(0))
