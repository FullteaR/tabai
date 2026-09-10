"""Tests for the Goldilocks-prime NTT multiplication path.

Layered from the ground up so a failure points at the smallest broken piece:

  1. constant self-checks      -- p, primitive root, roots of unity (no GPU)
  2. reference-NTT convolution  -- the Python oracle used by later GPU tests
  3. gl_mulmod kernel           -- the 128->64 bit modular reduction in isolation
  4. forward/inverse round-trip -- NTT stages compose to identity * n
  5. forward vs Python oracle   -- stage indexing matches §3.3 exactly
  6. _mul_ntt vs Python int     -- the whole path, incl. squaring / asymmetry
  7. large-size exactness       -- the regime where float FFT lost precision
  8. engine reuse               -- buffer/pad staleness across differing sizes

Tests 3-8 need a GPU (cupy + the NTT kernels in core.py); if the kernels are
not present yet they are skipped, so this file is committable in Phase 0.
"""

import random
import pytest

import sys
sys.set_int_max_str_digits(0)

# ---------------------------------------------------------------------------
# Goldilocks constants (mirror core.py / doc §3.1)
# ---------------------------------------------------------------------------
P = (1 << 64) - (1 << 32) + 1          # 0xFFFFFFFF00000001
PRIMITIVE_ROOT = 7
W32 = 0x185629DCDA58878C               # primitive 2^32-th root, pow(7,(P-1)>>32,P)


# ===========================================================================
# 1. Constant self-checks (no GPU)
# ===========================================================================
def _is_prime_mr(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2^64 (bases per doc §4 Phase 0)."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def test_p_is_prime():
    assert P == 0xFFFFFFFF00000001
    assert _is_prime_mr(P)


def test_seven_is_primitive_root():
    # p - 1 = 2^32 * (3 * 5 * 17 * 257 * 65537)
    assert (1 << 32) - 1 == 3 * 5 * 17 * 257 * 65537
    for q in (2, 3, 5, 17, 257, 65537):
        assert pow(PRIMITIVE_ROOT, (P - 1) // q, P) != 1
    assert pow(PRIMITIVE_ROOT, P - 1, P) == 1


def test_root_of_unity_order():
    w32 = pow(PRIMITIVE_ROOT, (P - 1) >> 32, P)
    assert w32 == W32
    assert pow(w32, 1 << 31, P) == P - 1   # order is exactly 2^32
    assert pow(w32, 1 << 32, P) == 1


# ===========================================================================
# 2. Reference NTT (Python oracle, doc §3.3)
# ===========================================================================
def ref_ntt_dif(a, w, p=P):
    """Natural-order in -> bit-reversed out (Gentleman-Sande)."""
    n = len(a)
    m = n
    while m >= 2:
        half = m // 2
        stride = n // m
        for base in range(0, n, m):
            for j in range(half):
                u = a[base + j]
                v = a[base + j + half]
                a[base + j] = (u + v) % p
                a[base + j + half] = (u - v) * w[j * stride] % p
        m = half


def ref_ntt_dit_inv(a, winv, p=P):
    """Bit-reversed in -> natural-order out (Cooley-Tukey)."""
    n = len(a)
    m = 2
    while m <= n:
        half = m // 2
        stride = n // m
        for base in range(0, n, m):
            for j in range(half):
                u = a[base + j]
                t = a[base + j + half] * winv[j * stride] % p
                a[base + j] = (u + t) % p
                a[base + j + half] = (u - t) % p
        m *= 2


def _next_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n


def ref_convolve(a_ch, b_ch, p=P):
    """Cyclic-safe linear convolution of two chunk lists via the reference NTT.
    Returns the first len(a)+len(b)-1 column sums (each an exact integer < p)."""
    n_conv = len(a_ch) + len(b_ch) - 1
    n = _next_pow2(n_conv)
    w_n = pow(PRIMITIVE_ROOT, (p - 1) // n, p)
    w_n_inv = pow(w_n, p - 2, p)
    inv_n = pow(n, p - 2, p)
    half = max(1, n // 2)
    w = [pow(w_n, k, p) for k in range(half)]
    winv = [pow(w_n_inv, k, p) for k in range(half)]
    fa = list(a_ch) + [0] * (n - len(a_ch))
    fb = list(b_ch) + [0] * (n - len(b_ch))
    ref_ntt_dif(fa, w, p)
    ref_ntt_dif(fb, w, p)
    fc = [x * y % p * inv_n % p for x, y in zip(fa, fb)]
    ref_ntt_dit_inv(fc, winv, p)
    return fc[:n_conv]


def _int_to_chunks16(x: int):
    ch = []
    while x:
        ch.append(x & 0xFFFF)
        x >>= 16
    return ch or [0]


def _chunks16_to_int(ch) -> int:
    x = 0
    for c in reversed(ch):
        x = (x << 16) + c
    return x


@pytest.mark.parametrize("seed", range(20))
def test_reference_convolution_matches_product(seed):
    rng = random.Random(seed)
    a = rng.getrandbits(rng.randint(1, 4000)) | 1
    b = rng.getrandbits(rng.randint(1, 4000)) | 1
    coeffs = ref_convolve(_int_to_chunks16(a), _int_to_chunks16(b))
    assert _chunks16_to_int(coeffs) == a * b


def test_reference_convolution_square():
    rng = random.Random(1234)
    a = rng.getrandbits(3000) | 1
    coeffs = ref_convolve(_int_to_chunks16(a), _int_to_chunks16(a))
    assert _chunks16_to_int(coeffs) == a * a


# ===========================================================================
# GPU-backed tests (3-8).  Skipped cleanly until the NTT kernels land.
# ===========================================================================
try:
    import cupy as cp
    import numpy as np
    from tabai_gpu import core as ntt_core
    from tabai_gpu.core import GPUBigInt, _BLOCK
    from tabai_gpu.utils import int_to_gpu, gpu_to_int
    _HAS_GPU = True
except Exception:  # pragma: no cover - environment without cupy
    _HAS_GPU = False


def _ntt_kernels_present() -> bool:
    return _HAS_GPU and hasattr(GPUBigInt, "_mul_ntt")


requires_ntt = pytest.mark.skipif(
    not _ntt_kernels_present(),
    reason="NTT kernels not implemented yet (added in Phase 1)",
)


@pytest.fixture
def calc():
    return GPUBigInt()


# ---- 3. gl_mulmod kernel in isolation -------------------------------------
# Drives the module-level pointwise kernel with scale=1, i.e. a[k]*b[k] mod p,
# so the 128->64 reduction is exercised on its own before any transform.
@requires_ntt
def test_gl_mulmod_kernel(calc):
    edge = [0, 1, 2, (1 << 32) - 1, 1 << 32, 1 << 63, P - 2, P - 1]
    rng = random.Random(0)
    vals = edge + [rng.randrange(P) for _ in range(400)]
    n = len(vals)
    a0 = np.array(vals, dtype=np.uint64)
    blocks = (n + _BLOCK - 1) // _BLOCK
    for bv in vals:
        a = cp.asarray(a0)
        b = cp.full(n, bv, dtype=cp.uint64)
        ntt_core._ntt_pointwise_scale_kernel(
            (blocks,), (_BLOCK,), (a, b, np.uint64(1), n))
        exp = np.array([(x * bv) % P for x in vals], dtype=np.uint64)
        assert np.array_equal(cp.asnumpy(a), exp), f"mismatch at b={bv}"


# ---- 4. forward/inverse round-trip ----------------------------------------
# forward (DIF, nat->bitrev) then inverse (DIT, bitrev->nat) is identity * n;
# scaling by inv_n via the pointwise kernel must recover the input exactly.
@requires_ntt
@pytest.mark.parametrize("logn", range(0, 13))
def test_forward_inverse_roundtrip(calc, logn):
    n = 1 << logn
    rng = random.Random(100 + logn)
    data = np.array([rng.randrange(P) for _ in range(n)], dtype=np.uint64)
    w_fwd, w_inv, inv_n = calc._get_ntt_tables(n)
    buf = cp.asarray(data)
    calc._ntt_forward(buf, n, w_fwd)
    calc._ntt_inverse(buf, n, w_inv)
    ones = cp.ones(n, dtype=cp.uint64)
    blocks = (n + _BLOCK - 1) // _BLOCK
    ntt_core._ntt_pointwise_scale_kernel(
        (blocks,), (_BLOCK,), (buf, ones, np.uint64(inv_n), n))
    assert np.array_equal(cp.asnumpy(buf), data)


# ---- 5. GPU forward vs Python oracle --------------------------------------
@requires_ntt
@pytest.mark.parametrize("n", [8, 64, 256, 512, 1024, 2048, 4096, 16384])
def test_forward_matches_reference(calc, n):
    rng = random.Random(7 + n)
    data = [rng.randrange(P) for _ in range(n)]
    w_fwd, _, _ = calc._get_ntt_tables(n)
    buf = cp.asarray(np.array(data, dtype=np.uint64))
    calc._ntt_forward(buf, n, w_fwd)
    w_n = pow(PRIMITIVE_ROOT, (P - 1) // n, P)
    w = [pow(w_n, k, P) for k in range(max(1, n // 2))]
    ref = list(data)
    ref_ntt_dif(ref, w)
    assert np.array_equal(cp.asnumpy(buf), np.array(ref, dtype=np.uint64))


@requires_ntt
@pytest.mark.parametrize("n", [256, 512, 1024, 2048, 8192])
def test_inverse_matches_reference_on_arbitrary_input(calc, n):
    # Check the inverse independently: matching forward/inverse indexing bugs
    # can cancel in a round-trip. Include canonical modular boundary values.
    rng = random.Random(871 + n)
    edge = [0, 1, P - 1, P - 2, (1 << 32) - 1, 1 << 32, 1 << 63]
    data = edge + [rng.randrange(P) for _ in range(n - len(edge))]
    _, w_inv, _ = calc._get_ntt_tables(n)
    # Guard an offset view to catch writes outside this transform's interval.
    guard = cp.full(n + 4, 123, dtype=cp.uint64)
    buf = guard[2:n + 2]
    buf[:] = cp.asarray(np.array(data, dtype=np.uint64))
    calc._ntt_inverse(buf, n, w_inv)
    root = pow(pow(PRIMITIVE_ROOT, (P - 1) // n, P), P - 2, P)
    expected = data.copy()
    ref_ntt_dit_inv(expected, [pow(root, k, P) for k in range(n // 2)])
    assert cp.asnumpy(buf).tolist() == expected
    assert cp.asnumpy(guard[:2]).tolist() == [123, 123]
    assert cp.asnumpy(guard[-2:]).tolist() == [123, 123]


# ---- 6. _mul_ntt vs Python int --------------------------------------------
def _mul_via_ntt(calc, a: int, b: int) -> int:
    ga, gb = int_to_gpu(a), int_to_gpu(b)
    return gpu_to_int(calc._mul_ntt(ga, gb))


@requires_ntt
@pytest.mark.parametrize("seed", range(25))
def test_mul_ntt_random(calc, seed):
    rng = random.Random(1000 + seed)
    a = rng.getrandbits(rng.randint(1, 200000)) | 1
    b = rng.getrandbits(rng.randint(1, 200000)) | 1
    assert _mul_via_ntt(calc, a, b) == a * b


@requires_ntt
@pytest.mark.parametrize("k", [8, 10, 12, 14])
def test_mul_ntt_pow2_boundaries(calc, k):
    # limb counts straddling a transform-length power of two (n_conv = 2^k -1,2^k,+1)
    for limbs in (((1 << k) // 4) - 1, (1 << k) // 4, ((1 << k) // 4) + 1):
        bits = limbs * 32
        a = (1 << bits) - 1        # all-ones: worst-case carries
        b = (1 << bits) - 1
        assert _mul_via_ntt(calc, a, b) == a * b


@requires_ntt
def test_mul_ntt_asymmetric(calc):
    rng = random.Random(55)
    small = rng.getrandbits(40) | 1
    big = rng.getrandbits(3_200_000) | 1     # ~100k limbs x 2 limbs
    assert _mul_via_ntt(calc, small, big) == small * big
    assert _mul_via_ntt(calc, big, small) == big * small


@requires_ntt
def test_mul_ntt_square_path(calc):
    rng = random.Random(77)
    a = rng.getrandbits(500000) | 1
    ga = int_to_gpu(a)
    # squaring path: same array object triggers is_square
    assert gpu_to_int(calc._mul_ntt(ga, ga, is_square=True)) == a * a


@requires_ntt
def test_mul_ntt_all_ones_stress(calc):
    for bits in (163872, 300000, 1_000_003):
        a = (1 << bits) - 1
        assert _mul_via_ntt(calc, a, a) == a * a


# ---- 7. large-size exactness (float FFT degraded to 8-bit chunks here) -----
@requires_ntt
@pytest.mark.parametrize("bits", [8_000_000, 64_000_000])
def test_mul_ntt_large_exact(calc, bits):
    rng = random.Random(bits)
    a = rng.getrandbits(bits) | 1
    b = rng.getrandbits(bits) | 1
    assert _mul_via_ntt(calc, a, b) == a * b


# ---- 8. engine reuse across differing sizes -------------------------------
@requires_ntt
def test_mul_ntt_engine_reuse(calc):
    rng = random.Random(999)
    sizes = [50000, 2000000, 130000, 3000000, 70000]
    for bits in sizes:
        a = rng.getrandbits(bits) | 1
        b = rng.getrandbits(bits) | 1
        assert _mul_via_ntt(calc, a, b) == a * b, f"failed at {bits} bits"
