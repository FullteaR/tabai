"""Exponentiation benchmark — TabaiInt vs gmpy2.mpz vs Python int.

Run standalone:
    python benchmarks/bench_pow.py
"""

from __future__ import annotations

import common
from common import (
    POW_BASE_BITS,
    POW_EXPONENTS,
    print_header,
    random_int,
    run_op_bench,
    standalone,
)


def run(backends: list[object]) -> None:
    print("pow  (base bits x exponent)")
    print_header(backends)
    # Track timeouts per exponent value independently.
    timed_out_by_exp: dict[int, set[str]] = {e: set() for e in POW_EXPONENTS}
    for base_bits in POW_BASE_BITS:
        for exp in POW_EXPONENTS:
            timed_out_by_exp[exp] |= run_op_bench(
                f"pow (exp={exp})", base_bits, backends,
                make_args=lambda b, e=exp: (random_int(b), e),
                run=lambda be, a, b: be.pow(a, b),
                skip_backends=timed_out_by_exp[exp],
            )
    print()


def main() -> None:
    common.random_seed()
    standalone(run)


if __name__ == "__main__":
    main()
