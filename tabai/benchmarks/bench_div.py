"""Division benchmark (floordiv and modulo) — TabaiInt vs gmpy2.mpz vs Python int.

Run standalone:
    python benchmarks/bench_div.py
"""

from __future__ import annotations

import common
from common import DIV_BIT_SIZES, print_header, random_int, run_op_bench, standalone


def run(backends: list[object]) -> None:
    # --- floor division -----------------------------------------------------
    print_header(backends)
    timed_out: set[str] = set()
    for bits in DIV_BIT_SIZES:
        timed_out |= run_op_bench(
            "floordiv", bits, backends,
            make_args=lambda b: (random_int(b), random_int(max(b // 2, 1))),
            run=lambda be, a, b: be.floordiv(a, b),
            skip_backends=timed_out,
        )
    print()

    # --- modulo -------------------------------------------------------------
    print_header(backends)
    timed_out = set()
    for bits in DIV_BIT_SIZES:
        timed_out |= run_op_bench(
            "mod", bits, backends,
            make_args=lambda b: (random_int(b), random_int(max(b // 2, 1))),
            run=lambda be, a, b: be.mod(a, b),
            skip_backends=timed_out,
        )
    print()


def main() -> None:
    common.random_seed()
    standalone(run)


if __name__ == "__main__":
    main()
