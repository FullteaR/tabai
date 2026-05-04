"""
Benchmark: TabaiInt (GPU) vs gmpy2.mpz vs Python int

Usage (inside the Docker container):
    python benchmarks/benchmark.py

Requirements:
    - cupy (included in the Docker image)
    - gmpy2: pip install gmpy2
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from typing import Callable

# ---------------------------------------------------------------------------
# Try importing each backend
# ---------------------------------------------------------------------------
try:
    from tabai_gpu import TabaiInt
    import cupy as cp

    HAS_TABAI = True
except ImportError:
    HAS_TABAI = False
    cp = None

try:
    import gmpy2

    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_GETRANDBITS_MAX = 1 << 29  # random.getrandbits is limited to ~2^31 on most platforms


def _random_int(bits: int) -> int:
    """Return a random positive integer with exactly *bits* bits."""
    if bits <= 0:
        return 0
    if bits <= _GETRANDBITS_MAX:
        return random.getrandbits(bits) | (1 << (bits - 1))
    # Build from chunks (MSB first so the shift arithmetic is simple).
    result = 0
    remaining = bits
    while remaining > 0:
        n = min(remaining, _GETRANDBITS_MAX)
        result = (result << n) | random.getrandbits(n)
        remaining -= n
    return result | (1 << (bits - 1))


def _gpu_sync() -> None:
    """Synchronize the default CUDA stream so GPU work is truly finished."""
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


_TIMEOUT_S = 10.0  # skip an operation if a single call exceeds this


def _bench(
    fn: Callable[[], object],
    warmup: int = 2,
    repeat: int = 5,
    timeout_s: float = _TIMEOUT_S,
) -> float | None:
    """Run *fn* with warm-up, return the **median** elapsed time in seconds.

    Returns *None* if any single call times out or raises an exception
    (e.g. GPU out-of-memory), so the caller can display a placeholder.
    """
    for _ in range(warmup):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        _gpu_sync()
        if time.perf_counter() - t0 > timeout_s:
            return None
    times: list[float] = []
    for _ in range(repeat):
        _gpu_sync()
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        _gpu_sync()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        if elapsed > timeout_s:
            return None
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Wrapper classes – uniform interface for each backend
# ---------------------------------------------------------------------------
class _PythonIntBackend:
    name = "Python int"

    @staticmethod
    def from_int(n: int) -> int:
        return n

    @staticmethod
    def add(a: int, b: int) -> int:
        return a + b

    @staticmethod
    def sub(a: int, b: int) -> int:
        return a - b

    @staticmethod
    def mul(a: int, b: int) -> int:
        return a * b

    @staticmethod
    def floordiv(a: int, b: int) -> int:
        return a // b

    @staticmethod
    def mod(a: int, b: int) -> int:
        return a % b

    @staticmethod
    def pow(a: int, b: int) -> int:
        return a ** b


class _Gmpy2Backend:
    name = "gmpy2.mpz"

    @staticmethod
    def from_int(n: int) -> gmpy2.mpz:
        return gmpy2.mpz(n)

    @staticmethod
    def add(a: gmpy2.mpz, b: gmpy2.mpz) -> gmpy2.mpz:
        return a + b

    @staticmethod
    def sub(a: gmpy2.mpz, b: gmpy2.mpz) -> gmpy2.mpz:
        return a - b

    @staticmethod
    def mul(a: gmpy2.mpz, b: gmpy2.mpz) -> gmpy2.mpz:
        return a * b

    @staticmethod
    def floordiv(a: gmpy2.mpz, b: gmpy2.mpz) -> gmpy2.mpz:
        return a // b

    @staticmethod
    def mod(a: gmpy2.mpz, b: gmpy2.mpz) -> gmpy2.mpz:
        return a % b

    @staticmethod
    def pow(a: gmpy2.mpz, b: gmpy2.mpz) -> gmpy2.mpz:
        return a ** b


class _TabaiBackend:
    name = "TabaiInt"

    @staticmethod
    def from_int(n: int) -> TabaiInt:
        return TabaiInt(n)

    @staticmethod
    def add(a: TabaiInt, b: TabaiInt) -> TabaiInt:
        return a + b

    @staticmethod
    def sub(a: TabaiInt, b: TabaiInt) -> TabaiInt:
        return a - b

    @staticmethod
    def mul(a: TabaiInt, b: TabaiInt) -> TabaiInt:
        return a * b

    @staticmethod
    def floordiv(a: TabaiInt, b: TabaiInt) -> TabaiInt:
        return a // b

    @staticmethod
    def mod(a: TabaiInt, b: TabaiInt) -> TabaiInt:
        return a % b

    @staticmethod
    def pow(a: TabaiInt, b: TabaiInt) -> TabaiInt:
        return a ** b


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
BIT_SIZES     = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
                 1_000_000_000, 10_000_000_000]

DIV_BIT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
                 1_000_000_000]

# For pow the exponent must be small; otherwise all backends are impractical.
POW_EXPONENTS = [2, 3, 10, 20]
POW_BASE_BITS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "   >timeout"
    if seconds < 1e-3:
        return f"{seconds * 1e6:>10.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:>10.2f} ms"
    return f"{seconds:>10.3f}  s"


def _print_header(backends: list[object]) -> None:
    names = [b.name for b in backends]
    col = 14
    header = f"{'Operation':<28} {'bits':>10}"
    for n in names:
        header += f" {n:>{col}}"
    print(header)
    print("-" * len(header))


def _run_op_bench(
    op_name: str,
    bits: int,
    backends: list[object],
    make_args: Callable,
    run: Callable,
    skip_backends: set[str] | None = None,
    warmup: int = 2,
    repeat: int = 5,
    timeout_s: float = _TIMEOUT_S,
) -> set[str]:
    """Run one benchmark row. Returns the set of backend names that timed out
    (or OOM'd) at this bit size so callers can skip them for larger sizes."""
    col = 14
    row = f"{op_name:<28} {bits:>10}"
    newly_timed_out: set[str] = set()
    for backend in backends:
        if skip_backends and backend.name in skip_backends:
            row += f" {'--':>{col}}"
            continue
        args_raw = make_args(bits)
        args = tuple(backend.from_int(x) for x in args_raw)
        elapsed = _bench(
            lambda a=args, r=run, be=backend: r(be, *a),
            warmup=warmup,
            repeat=repeat,
            timeout_s=timeout_s,
        )
        if elapsed is None:
            newly_timed_out.add(backend.name)
        row += f" {_format_time(elapsed):>{col}}"
    print(row, flush=True)
    return newly_timed_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(42)

    backends: list[object] = []
    if HAS_TABAI:
        backends.append(_TabaiBackend())
    if HAS_GMPY2:
        backends.append(_Gmpy2Backend())
    backends.append(_PythonIntBackend())

    if not backends:
        print("ERROR: no backends available", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("  tabai benchmark  –  multi-precision integer performance comparison")
    print("=" * 72)
    print(f"\nBackends: {', '.join(b.name for b in backends)}")
    print(f"Bit sizes: {BIT_SIZES}")
    print()

    # --- addition -----------------------------------------------------------
    _print_header(backends)
    timed_out: set[str] = set()
    for bits in BIT_SIZES:
        timed_out |= _run_op_bench(
            "add", bits, backends,
            make_args=lambda b: (_random_int(b), _random_int(b)),
            run=lambda be, a, b: be.add(a, b),
            skip_backends=timed_out,
        )
    print()

    # --- subtraction --------------------------------------------------------
    _print_header(backends)
    timed_out = set()
    for bits in BIT_SIZES:
        timed_out |= _run_op_bench(
            "sub (a > b)", bits, backends,
            make_args=lambda b: (_random_int(b) | (1 << b), _random_int(b)),
            run=lambda be, a, b: be.sub(a, b),
            skip_backends=timed_out,
        )
    print()

    # --- multiplication -----------------------------------------------------
    _print_header(backends)
    timed_out = set()
    for bits in BIT_SIZES:
        timed_out |= _run_op_bench(
            "mul", bits, backends,
            make_args=lambda b: (_random_int(b), _random_int(b)),
            run=lambda be, a, b: be.mul(a, b),
            skip_backends=timed_out,
        )
    print()

    # --- floor division -----------------------------------------------------
    _print_header(backends)
    timed_out = set()
    for bits in DIV_BIT_SIZES:
        timed_out |= _run_op_bench(
            "floordiv", bits, backends,
            make_args=lambda b: (_random_int(b), _random_int(max(b // 2, 1))),
            run=lambda be, a, b: be.floordiv(a, b),
            skip_backends=timed_out,
        )
    print()

    # --- modulo -------------------------------------------------------------
    _print_header(backends)
    timed_out = set()
    for bits in DIV_BIT_SIZES:
        timed_out |= _run_op_bench(
            "mod", bits, backends,
            make_args=lambda b: (_random_int(b), _random_int(max(b // 2, 1))),
            run=lambda be, a, b: be.mod(a, b),
            skip_backends=timed_out,
        )
    print()

    # --- pow ----------------------------------------------------------------
    print("pow  (base bits x exponent)")
    _print_header(backends)
    # Track timeouts per exponent value independently.
    timed_out_by_exp: dict[int, set[str]] = {e: set() for e in POW_EXPONENTS}
    for base_bits in POW_BASE_BITS:
        for exp in POW_EXPONENTS:
            timed_out_by_exp[exp] |= _run_op_bench(
                f"pow (exp={exp})", base_bits, backends,
                make_args=lambda b, e=exp: (_random_int(b), e),
                run=lambda be, a, b: be.pow(a, b),
                skip_backends=timed_out_by_exp[exp],
            )
    print()

    print("Done.")


if __name__ == "__main__":
    main()
