"""Subtraction benchmark — TabaiInt vs gmpy2.mpz vs Python int.

Run standalone:
    python benchmarks/bench_sub.py
"""

from __future__ import annotations

import common
from common import BIT_SIZES, print_header, random_int, run_op_bench, standalone


def run(backends: list[object]) -> None:
    print_header(backends)
    timed_out: set[str] = set()
    for bits in BIT_SIZES:
        timed_out |= run_op_bench(
            "sub (a > b)", bits, backends,
            make_args=lambda b: (random_int(b) | (1 << b), random_int(b)),
            run=lambda be, a, b: be.sub(a, b),
            skip_backends=timed_out,
        )
    print()


def main() -> None:
    common.random_seed()
    standalone(run)


if __name__ == "__main__":
    main()
