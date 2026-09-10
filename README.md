# tabai

**GPU-accelerated arbitrary-precision integers for Python.**
`tabai` runs custom CUDA kernels through CuPy so that big integers live on the GPU, and exposes a single
`TabaiInt` class that behaves like Python's built-in `int`. Huge multiplications use an NTT (number-theoretic
transform) and can be split across several GPUs.

[日本語版 README](README_ja.md)

## Supported operations

| Operation | API |
| --- | --- |
| Create from / convert to Python `int` | `TabaiInt(value)` / `value.to_cpu()` |
| Add, subtract, multiply | `+`, `-`, `*` |
| Floor division, modulo | `//`, `%`, `divmod(a, b)` |
| Power (non-negative exponent) | `**`, `pow(a, b)` |
| Negation, absolute value | `-a`, `abs(a)` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |

`TabaiInt` can be mixed with Python `int` in any of these. Division follows Python semantics (floor toward
negative infinity, remainder takes the divisor's sign).

## Install

Requirements: an NVIDIA GPU with drivers, Docker Compose, and GPU access from Docker
(e.g. the NVIDIA Container Toolkit).

```bash
git clone https://github.com/FullteaR/tabai.git
cd tabai
docker compose up -d --build
docker compose exec tabai bash
```

The container is built on `cupy/cupy:v14.0.1` and mounts `./tabai` at `/mnt` with `/mnt/src` on
`PYTHONPATH`, so `import tabai_gpu` works immediately. There is no pip package yet.

If GPU initialization fails with `cudaErrorCompatNotSupportedOnDevice`, add the host CUDA libraries
to the container environment:

```bash
docker compose exec -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai bash
```

## Usage

```python
from tabai_gpu import TabaiInt

a = TabaiInt((1 << 4096) - 1)
b = TabaiInt((1 << 2048) + 1)

product = a * b
print(product.to_cpu())          # back to a Python int

q, r = divmod(TabaiInt(-7), 3)   # (-3, 2) — same as Python
big = a ** 3
mixed = a + 1                    # Python ints are accepted directly
assert mixed > a
```

### Multiple GPUs

Inside a `multi_gpu` scope, large NTT multiplications are sharded across the given devices
(this also covers the multiplications inside `**`, `//` and `%`):

```python
from tabai_gpu import TabaiInt, multi_gpu

a = TabaiInt((1 << 200_000_000) - 1)
b = TabaiInt((1 << 200_000_000) + 7)

with multi_gpu([0, 1]):
    product = a * b
```

By default, products below 64 Mbit stay on a single GPU; pass `min_bits=0` to distribute smaller ones too.
See [docs/multi-gpu.md](docs/multi-gpu.md) for details.

## Performance

Measured on 2 × RTX 3090 (2026-09-10), median of 7 runs, times in ms:

| Operation | Input bits | tabai (1 GPU) | tabai (2 GPUs) | GMP (CPU) |
| --- | ---: | ---: | ---: | ---: |
| Multiply | 100,000,000 | 36.3 | 32.0 | 873.3 |
| Square | 100,000,000 | 25.2 | 25.7 | 589.3 |
| Cube (`** 3`) | 100,000,000 | 104.8 | 91.2 | 1980.1 |
| Multiply | 512,000,000 | 176.0 | 146.6 | — |

GPU kernel launch, synchronization and transfer costs mean small inputs are often faster on the CPU.
Full results: [docs/benchmarks/rtx3090-2026-09-10.md](docs/benchmarks/rtx3090-2026-09-10.md).

## Tests and benchmarks

```bash
# tests
docker compose exec tabai pytest tests/ -q

# all operations, single GPU vs GMP vs Python int
docker compose exec -T tabai python benchmarks/benchmark.py \
    --gpu-mode single --backends tabai gmpy2 python --bits 1000000 --pow-exponents 3

# multiplication only, single vs two GPUs
docker compose exec -T tabai python benchmarks/bench_mul.py \
    --gpu-mode both --devices 0 1 --backends tabai --bits 100000000
```

The CPU-only backends need neither CUDA nor Docker (`gmpy2` must be installed for that backend):

```bash
python3 tabai/benchmarks/benchmark.py --backends python gmpy2 --bits 1000 10000
```

All options and output formats are documented in [docs/benchmarks.md](docs/benchmarks.md).

## How it works

| File | Role |
| --- | --- |
| [tabai_int.py](tabai/src/tabai_gpu/tabai_int.py) | Public `TabaiInt` class, sign handling, operators, `int` interop |
| [core.py](tabai/src/tabai_gpu/core.py) | `GPUBigInt` and the CUDA kernels, buffer and NTT-table reuse |
| [multi_gpu.py](tabai/src/tabai_gpu/multi_gpu.py) | Sharded NTT across GPUs and the `multi_gpu` context manager |
| [utils.py](tabai/src/tabai_gpu/utils.py) | Conversion between Python `int` and GPU limb arrays |

Magnitudes are stored on the GPU as little-endian `uint32` limb arrays; the sign is tracked in Python.

- **Add / subtract** — parallel prefix scan propagates carries and borrows.
- **Multiply** — schoolbook kernel for small operands, otherwise an NTT over the Goldilocks prime
  `p = 2^64 − 2^32 + 1`. All arithmetic is modular, so there is no floating-point rounding.
- **Divide / modulo** — Newton–Raphson reciprocal, then a ±1 quotient correction.
- **Power** — sliding-window exponentiation driven from the CPU, with GPU multiplies and squarings.

Not implemented on the public class: `/`, bitwise and shift operators, three-argument
`pow(a, b, mod)`, and `int(a)` — use `to_cpu()` to get a Python `int`.

## Documentation

- [docs/multi-gpu.md](docs/multi-gpu.md) — multi-GPU algorithm, configuration and validation
- [docs/benchmarks.md](docs/benchmarks.md) — benchmark CLI reference
- [docs/performance.md](docs/performance.md) — optimization report
- [CLAUDE.md](CLAUDE.md) — development notes and implementation constraints

## License

[MIT License](LICENSE)
