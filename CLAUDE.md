# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`tabai` is a GPU-accelerated arbitrary-precision integer library for Python. It implements big integer arithmetic using CUDA kernels via [CuPy](https://cupy.dev/), with `TabaiInt` as the public-facing Python class.

## Development Environment

All development happens inside a Docker container with GPU access. The container mounts `./tabai` as `/mnt` and sets `PYTHONPATH` to `/mnt/src`.

```bash
# Build and start the container
docker compose up -d

# Attach to the running container
docker compose exec tabai bash

# Inside the container — run all tests
pytest tests/

# Run a specific test file
pytest tests/test_tabai_int.py

# Run a single test
pytest tests/test_tabai_int.py::test_add_basic

# Run the full benchmark suite (all operations)
python benchmarks/benchmark.py

# Run a single operation's benchmark
python benchmarks/bench_add.py   # also: bench_sub, bench_mul, bench_div, bench_pow
```

The benchmarks live under `benchmarks/`. Shared infrastructure (random-input
generation, timing, per-backend wrappers, table printing, bit-size config) is in
`common.py`; each operation has its own `bench_*.py` exposing `run(backends)` and
a standalone `main()`; `benchmark.py` runs them all in sequence. To add an
operation, create a new `bench_*.py` and register it in `benchmark.py`'s
`_BENCHMARKS` list.

## Architecture

### Layer 1 — GPU kernels (`tabai/src/tabai_gpu/core.py`)

`GPUBigInt` holds all CUDA logic. Big integers are represented as little-endian arrays of `uint32` limbs on the GPU. Key design choices:

- **Add/Sub**: parallel prefix scan (`compute_states` → `block_scan` → `propagate` → `apply_carries`). The scan propagates carry/borrow across limbs without sequential chaining.
- **Mul**: dispatch on per-column work `la*lb`. Small operands (`la*lb ≤ _MUL_SCHOOLBOOK_MAX_WORK`, ~6016 limbs squared) use an all-GPU schoolbook kernel (`schoolbook_mul16` — base-2^16 column sums). Larger operands use an **NTT** (number-theoretic transform) over the Goldilocks prime `p = 2^64 − 2^32 + 1`: `uint16` chunks, radix-2 DIF forward / DIT inverse butterflies (no bit-reversal pass), per-size cached twiddle tables, and a fused pointwise-product + 1/n scaling kernel. All arithmetic is mod `p` (128→64 bit reduction via `2^64 ≡ 2^32−1`, `2^96 ≡ −1`), so the convolution is **exact at any size** — no float rounding. Squaring (`a_gpu is b_gpu`) runs a single forward transform. Both paths emit base-2^16 column sums into the same on-GPU carry-resolution pipeline (fixed-count carry-reduction loop + the add/sub parallel prefix scan), with zero GPU→CPU syncs.
- **Div/Mod**: Newton–Raphson reciprocal. `_reciprocal` computes `floor(2^p / b)`; for large `p` it runs a doubling-precision GPU Newton iteration (`_newton_reciprocal`, ~O(log n) full-size muls), below `_DIV_NEWTON_THRESHOLD_LIMBS` it falls back to CPU big-int `//`. `divmod` then forms `q0 = floor(a·x / 2^p)` (a free limb slice) and corrects `q0` by at most ±1. No O(n) GPU round-trip loop.
- **Compare**: parallel `atomicMax` encodes the highest differing index and which side is larger into a single `uint64`.

### Layer 2 — Python integer wrapper (`tabai/src/tabai_gpu/tabai_int.py`)

`TabaiInt` wraps `GPUBigInt` and adds sign tracking (`_sign = ±1`). A module-level singleton `_shared_gpu_big_int = GPUBigInt()` is reused for all operations. `TabaiInt` supports interop with Python `int` via `_coerce()`.

### Layer 3 — Utilities (`tabai/src/tabai_gpu/utils.py`)

`int_to_gpu` / `gpu_to_int`: convert between Python `int` and `uint32` CuPy arrays using little-endian byte representation.

### Public API (`tabai/src/tabai_gpu/__init__.py`)

Only `TabaiInt` is exported.

## Key Constraints

- `__pow__` uses left-to-right sliding-window exponentiation (window size `k` scales 1→6 with exponent bit length; precomputes odd powers of the base). The exponent is scanned on CPU, squarings/muls run via `GPUBigInt.mul`. Negative exponents raise `ValueError`.
- `__divmod__` adjusts quotient/remainder to match Python's floor-division semantics (result sign follows the divisor).
- `GPUBigInt.mul` is exact at every size: the large path is a mod-`p` NTT (integer arithmetic, no rounding). Its ceiling is memory, not precision — the int64 carry pipeline stays valid while the shorter operand is under ~2^31 limbs (~32 Gbit), asserted at the top of `_mul_ntt`.
- The `_trim` kernel uses `atomicMax` to find the last non-zero limb — always call `_trim` after any operation that may leave leading zero limbs.
