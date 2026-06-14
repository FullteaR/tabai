"""
Benchmark runner: TabaiInt (GPU) vs gmpy2.mpz vs Python int

Runs every per-operation benchmark in sequence. To run just one operation,
execute its module directly, e.g.:

    python benchmarks/bench_add.py
    python benchmarks/bench_mul.py

Usage (inside the Docker container):
    python benchmarks/benchmark.py

Requirements:
    - cupy (included in the Docker image)
    - gmpy2: pip install gmpy2
"""

from __future__ import annotations

import sys

import common
from common import BIT_SIZES, available_backends

import bench_add
import bench_sub
import bench_mul
import bench_div
import bench_pow

# Each entry is (banner title, module). The module exposes ``run(backends)``.
_BENCHMARKS = [
    ("addition", bench_add),
    ("subtraction", bench_sub),
    ("multiplication", bench_mul),
    ("division (floordiv / mod)", bench_div),
    ("exponentiation", bench_pow),
]


def main() -> None:
    common.random_seed()

    backends = available_backends()
    if not backends:
        print("ERROR: no backends available", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("  tabai benchmark  –  multi-precision integer performance comparison")
    print("=" * 72)
    print(f"\nBackends: {', '.join(b.name for b in backends)}")
    print(f"Bit sizes: {BIT_SIZES}")
    print()

    for title, module in _BENCHMARKS:
        print(f"### {title}")
        module.run(backends)

    print("Done.")


if __name__ == "__main__":
    main()
