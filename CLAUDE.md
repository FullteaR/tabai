# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`tabai` is a GPU-accelerated arbitrary-precision integer library for Python. It implements big integer arithmetic using CUDA kernels via CuPy, with `TabaiInt` as the public-facing Python class.

See [README.md](README.md) for supported operations, usage, environment requirements, and benchmark interpretation. The benchmarks compare against Python `int` and `gmpy2.mpz`; GPU acceleration is not a speed guarantee for every input size.

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

The benchmark entry points share `runner.py` (CLI, sequential spawned backend workers,
per-stage timeouts, output) and `common.py` (deterministic cases, lazy backend adapters,
timing, CPU-reference checks). Every script supports `--gpu-mode single|multi|both`,
`--devices`, `--backends`, `--bits`, and the same timing/output flags. See
[docs/benchmarks.md](docs/benchmarks.md). Direct scripts and `python -m benchmarks.*`
are supported. Add operations to `common.OPERATIONS` / `OP_FUNCTIONS`, input generation,
and `runner.cases_for`; the small entry scripts only select defaults.

Keep CUDA imports/initialization inside GPU backend workers. Generate each case from
its own seed so every backend gets the same input, and keep preparation/verification
outside the timing interval. Never relabel arbitrary exceptions as timeouts. A failed
worker must be stopped before another backend or retry starts. Benchmark regression
tests are in `tests/test_benchmarks.py` and can run without CUDA.

## Architecture

### Layer 1 — GPU kernels (`tabai/src/tabai_gpu/core.py`)

`GPUBigInt` holds all CUDA logic. Big integers are represented as little-endian arrays of `uint32` limbs on the GPU. Key design choices:

- **Add/Sub**: parallel prefix scan (`compute_states` → `block_scan` → `propagate` → `apply_carries`). The scan propagates carry/borrow across limbs without sequential chaining.
- **Mul**: dispatch on per-column work `la*lb`. Small operands (`la*lb ≤ _MUL_SCHOOLBOOK_MAX_WORK`, ~6016 limbs squared) use an all-GPU schoolbook kernel (`schoolbook_mul16` — base-2^16 column sums). Larger operands use an **NTT** (number-theoretic transform) over the Goldilocks prime `p = 2^64 − 2^32 + 1`: `uint16` chunks, DIF forward / DIT inverse butterflies (no bit-reversal pass), two global stages fused in registers plus up to 1024 coefficients fused in shared memory, per-size cached twiddle tables, and a fused pointwise-product + 1/n scaling kernel. All arithmetic is mod `p` (128→64 bit reduction via `2^64 ≡ 2^32−1`, `2^96 ≡ −1`), so the convolution is **exact within the coefficient and transform-length bounds below** — no float rounding. Squaring (`a_gpu is b_gpu`) runs a single forward transform. Both single-device paths emit base-2^16 column sums into the same on-GPU carry-resolution pipeline (fixed-count carry-reduction loop + the add/sub parallel prefix scan), with zero GPU→CPU syncs.
- **Div/Mod**: Newton–Raphson reciprocal. `_reciprocal` computes `floor(2^p / b)`; for large `p` it runs a doubling-precision GPU Newton iteration (`_newton_reciprocal`, ~O(log n) full-size muls), below `_DIV_NEWTON_THRESHOLD_LIMBS` it falls back to CPU big-int `//`. `divmod` then forms `q0 = floor(a·x / 2^p)` (a free limb slice) and corrects `q0` by at most ±1. No O(n) GPU round-trip loop.
- **Compare**: parallel `atomicMax` encodes the highest differing index and which side is larger into a single `uint64`.

### Layer 2 — Python integer wrapper (`tabai/src/tabai_gpu/tabai_int.py`)

`TabaiInt` wraps `GPUBigInt` and adds sign tracking (`_sign = ±1`). `_get_engine()` lazily reuses a `GPUBigInt` per CUDA device and stream. Scalar constants also belong to each engine. `TabaiInt` supports interop with Python `int` via `_coerce()`.

### Layer 3 — Utilities (`tabai/src/tabai_gpu/utils.py`)

`int_to_gpu` / `gpu_to_int`: convert between Python `int` and `uint32` CuPy arrays using little-endian byte representation.

### Public API (`tabai/src/tabai_gpu/__init__.py`)

`TabaiInt` and the `multi_gpu(devices, min_bits=64_000_000, transfer="auto")` context manager are exported. See [docs/multi-gpu.md](docs/multi-gpu.md) for the sharded DIF/DIT algorithm, ownership/synchronization rules, and physical two-GPU validation commands.

## Key Constraints

- `__pow__` uses left-to-right sliding-window exponentiation (window size `k` scales 1→6 with exponent bit length; precomputes odd powers of the base). The exponent is scanned on CPU, squarings/muls run via `GPUBigInt.mul`. Negative exponents raise `ValueError`.
- `__divmod__` adjusts quotient/remainder to match Python's floor-division semantics (the quotient rounds toward negative infinity; a nonzero remainder has the divisor's sign).
- NTT multiplication avoids floating-point rounding within its coefficient and transform-length bounds. `_mul_ntt` asserts `min(n_a, n_b) * 65535**2 < 2**63`, where `n_a` and `n_b` count **16-bit chunks**, not 32-bit limbs (roughly 2^31 chunks, or 32 Gbit). The Goldilocks prime supports power-of-two transform lengths up to 2^32; the implementation does not explicitly check that length bound. GPU memory is also a practical limit.
- Device/stream-local engines allocate GPU arrays lazily and reuse mutable scratch buffers and cached NTT tables. There is no library-wide CPU-only fallback. `to_cpu()` and `repr()` transfer the value to the host.
- The public wrapper does not implement true division, bitwise/shift operators, three-argument modular `pow`, or `__int__`; use `to_cpu()` for conversion.
- The `_trim` kernel uses `atomicMax` to find the last non-zero limb — always call `_trim` after any operation that may leave leading zero limbs.

## Distributed multiplication

`core._distributed_multiplier` is a `ContextVar` set by `multi_gpu`. `GPUBigInt.mul` keeps schoolbook dispatch first, then delegates eligible NTT products to that executor; internal pow/division products follow the same route. The executor partitions contiguous NTT coefficient intervals over a power-of-two number of devices, exchanges partner snapshots for cross-rank stages, and gathers coefficients to the primary engine's existing carry pipeline. Rank streams/buffers are independent. P2P copies use explicit streams; unavailable links use reusable pinned host staging, pipelined in 2 MiB chunks for large contiguous copies. The primary device, input arrays, and calling engine must agree.

Do not treat logical ranks on one device as evidence of physical multi-GPU correctness or performance. `pytest tests/test_multi_gpu.py --require-multi-gpu -q` requires at least two visible CUDA devices; `benchmarks/bench_multi_gpu.py` compares identical inputs in fresh processes and records transfer counts and timing. The default 64 Mbit threshold was chosen after measuring two RTX 3090s with host staging: below it transfer overhead outweighed the saved computation. It remains configurable for other topologies. Two-device host-staged execution is validated; physical P2P and four-device execution are not validated on this VM.

For two ranks, when an operand fits entirely in the lower input half, `_prepare_forward` distributes the uint16 input and constructs `u` / `u*w` directly, avoiding a uint64 exchange of zero padding. `_inverse_two` gathers only the upper rank and fuses the last inverse butterfly into the primary carry buffer. Longer asymmetric operands use the generic forward exchange; more than two ranks use generic forward/inverse exchange. CPU staging buffers cannot be overwritten or grown until their previous upload event completes. Dedicated download streams overlap subsequent D2H chunks with prior H2D uploads; each source chunk must be complete before upload. Drain download and rank streams on failure/close. Single-device NTT carry resolution uses only the linear convolution plus a carry-out slot, excluding transform padding.

See [docs/performance.md](docs/performance.md) for the 2026-09-10 profile and verified before/after measurements. The original 64 Mbit distribution threshold remains configurable and is not a speed guarantee after kernel changes.
