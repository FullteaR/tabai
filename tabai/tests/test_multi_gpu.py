"""Real CUDA rank tests on one card, plus explicitly gated two-card tests."""
import random

import cupy as cp
import numpy as np
import pytest

from tabai_gpu import TabaiInt, multi_gpu
from tabai_gpu import core
from tabai_gpu.multi_gpu import _DistributedMultiplier, _layout
from tabai_gpu.tabai_int import _get_engine
from tabai_gpu.utils import int_to_gpu, gpu_to_int
from .test_ntt import P, ref_ntt_dif


DEVICES = [
    pytest.param((0,), id="one-rank"),
    pytest.param((0, 0), id="two-logical-ranks"),
    pytest.param((0, 0, 0, 0), id="four-logical-ranks"),
    pytest.param((0, 1), id="two-physical-gpus", marks=pytest.mark.skipif(
        cp.cuda.runtime.getDeviceCount() < 2, reason="requires two physical GPUs")),
]


@pytest.fixture(params=DEVICES)
def devices(request):
    return request.param


@pytest.fixture(params=["auto", "host"])
def executor(devices, request):
    with cp.cuda.Device(0):
        calc = _DistributedMultiplier(devices, min_bits=0, transfer=request.param)
        try:
            yield calc
        finally:
            calc.close()


def test_distributed_forward_and_inverse(executor):
    rng = random.Random(741)
    for n in [len(executor.ranks), 2 * len(executor.ranks), 256]:
        size = n // len(executor.ranks)
        values = [rng.randrange(P) for _ in range(n)]
        for i, rank in enumerate(executor.ranks):
            with cp.cuda.Device(rank.device_id), rank.stream:
                rank.ensure(size)
                rank.a[:size] = cp.asarray(np.array(values[i * size:(i + 1) * size], dtype=np.uint64))
        executor._forward("a", n, size)
        executor.synchronize()
        actual = np.concatenate([rank.a[:size].get() for rank in executor.ranks])
        root = pow(7, (P - 1) // n, P)
        expected = values.copy()
        ref_ntt_dif(expected, [pow(root, k, P) for k in range(max(1, n // 2))])
        assert actual.tolist() == expected
        executor._inverse("a", n, size)
        executor.synchronize()
        actual = np.concatenate([rank.a[:size].get() for rank in executor.ranks])
        inv_n = pow(n, P - 2, P)
        assert [int(v) * inv_n % P for v in actual] == values


def test_products_carries_boundaries_and_buffer_reuse(executor):
    engine = core.GPUBigInt()
    rng = random.Random(84)
    results = []
    pairs = [(0, 0), (0, 17), (1, 1), (0xFFFFFFFF, 0xFFFFFFFF)]
    for bits in [31, 32, 33, 511, 512, 513, 20_003, 200_000, 63]:
        pairs.extend([
            ((1 << bits) - 1, (1 << bits) - 1),
            (rng.getrandbits(bits) | 1, rng.getrandbits(bits + 33) | 1),
        ])
    pairs.extend([(17, (1 << 200_003) - 1), ((1 << 200_003) - 1, 17)])
    for a, b in pairs:
        ga, gb = int_to_gpu(a), int_to_gpu(b)
        product = executor.mul(engine, ga, gb)
        assert gpu_to_int(product) == a * b
        results.append((product, a * b))
        assert gpu_to_int(ga) == a and gpu_to_int(gb) == b
    # Results must survive all subsequent writes to the executor's scratch.
    for product, expected in results:
        assert gpu_to_int(product) == expected


def test_square_and_noncontiguous_inputs(executor):
    engine = core.GPUBigInt()
    value = (1 << 200_017) - 1
    ga = int_to_gpu(value)
    assert gpu_to_int(executor.mul(engine, ga, ga)) == value * value
    padded = cp.zeros(2 * len(ga), dtype=cp.uint32)
    padded[::2] = ga
    assert gpu_to_int(executor.mul(engine, padded[::2], ga)) == value * value


def test_public_arithmetic_and_context_restore(devices, monkeypatch):
    # Duplicate IDs are for internal rank tests only; here test the public API
    # with one card or two distinct physical cards.
    if len(set(devices)) != len(devices):
        pytest.skip("logical ranks use the internal executor")
    monkeypatch.setattr(core, "_MUL_SCHOOLBOOK_MAX_WORK", 0)
    a, b = -((1 << 200_001) - 17), (1 << 100_003) + 3
    with multi_gpu(devices, min_bits=0) as executor:
        ga, gb = TabaiInt(a), TabaiInt(b)
        assert (ga * gb).to_cpu() == a * b
        assert (ga ** 3).to_cpu() == a ** 3
        q, r = divmod(ga, gb)
        assert (q.to_cpu(), r.to_cpu()) == divmod(a, b)
        assert (ga + gb).to_cpu() == a + b
        assert (ga - gb).to_cpu() == a - b
        assert executor.multiplications > 3
        with pytest.raises(RuntimeError, match="intentional"):
            with multi_gpu(devices, min_bits=0):
                raise RuntimeError("intentional")
        assert core._distributed_multiplier.get() is executor
    assert core._distributed_multiplier.get() is None
    assert (ga * gb).to_cpu() == a * b


def test_threshold_keeps_small_operations_local():
    with multi_gpu([0], min_bits=10_000_000) as executor:
        a = TabaiInt((1 << 250_000) - 1)  # NTT, below the distributed threshold
        assert (a * a).to_cpu() == ((1 << 250_000) - 1) ** 2
        assert executor.multiplications == 0
    with multi_gpu([0], min_bits=0) as executor:
        assert (TabaiInt(7) * 9).to_cpu() == 63  # schoolbook wins dispatch first
        assert executor.multiplications == 0


def test_nondefault_stream(executor):
    with cp.cuda.Stream(non_blocking=True):
        engine = core.GPUBigInt()
        a, b = (1 << 20_003) - 1, (1 << 19_999) + 7
        ga, gb = int_to_gpu(a), int_to_gpu(b)
        assert gpu_to_int(executor.mul(engine, ga, gb)) == a * b


def test_engines_are_stream_local():
    first = _get_engine()
    with cp.cuda.Stream(non_blocking=True):
        second = _get_engine()
        assert second is _get_engine() and second is not first
        assert (TabaiInt(-13) // 5).to_cpu() == -3
    assert _get_engine() is first


@pytest.mark.skipif(cp.cuda.runtime.getDeviceCount() < 2, reason="requires two physical GPUs")
def test_second_gpu_primary_and_device_restore(monkeypatch):
    monkeypatch.setattr(core, "_MUL_SCHOOLBOOK_MAX_WORK", 0)
    original = cp.cuda.runtime.getDevice()
    first = _get_engine()
    with cp.cuda.Device(1), multi_gpu([1, 0], min_bits=0) as executor:
        second = _get_engine()
        assert second is not first and second._one.device.id == 1
        a = TabaiInt(-((1 << 4097) - 1))
        b = TabaiInt((1 << 2051) + 7)
        assert (a * b).to_cpu() == a.to_cpu() * b.to_cpu()
        assert (a // b).to_cpu() == a.to_cpu() // b.to_cpu()
        assert cp.cuda.runtime.getDevice() == 1
        assert executor.peer_copies + executor.host_copies > 0
    assert cp.cuda.runtime.getDevice() == original


@pytest.mark.parametrize("kwargs", [
    {"devices": []}, {"devices": [0, 0]}, {"devices": [0, 0, 0]},
    {"devices": [-1]}, {"devices": [cp.cuda.runtime.getDeviceCount()]},
    {"devices": [0], "min_bits": -1}, {"devices": [0], "transfer": "invalid"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        with multi_gpu(**kwargs):
            pytest.fail("invalid configuration accepted")
    assert core._distributed_multiplier.get() is None


def test_layout_limits_without_large_allocations():
    assert _layout(1, 1, 2) == (2, 2, 3, 4, 2)
    for args in [(0, 1, 2), (1, 1, 3), (1 << 30, 1 << 30, 2),
                 (1, 1 << 32, 2), (1, 1 << 30, 4)]:
        with pytest.raises(ValueError):
            _layout(*args)


def test_large_carry_chain_across_rank_boundary(executor):
    # Multi-megabyte transfers and long carries across the rank boundary, with
    # a closed-form integer oracle that avoids a costly CPU multiplication.
    bits = 8_000_003
    value = (1 << bits) - 1
    a = int_to_gpu(value)
    engine = core.GPUBigInt()
    actual = executor.mul(engine, a, a)
    expected = (1 << (2 * bits)) - (1 << (bits + 1)) + 1
    assert gpu_to_int(actual) == expected
    assert gpu_to_int(a) == value
    assert cp.cuda.runtime.getDevice() == 0


@pytest.mark.parametrize("destination", [
    0,
    pytest.param(1, marks=pytest.mark.skipif(
        cp.cuda.runtime.getDeviceCount() < 2, reason="requires two physical GPUs")),
])
def test_pinned_upload_reuse_across_streams_and_growth(destination):
    # Consecutive copies share a staging slot but have different destination
    # streams. Reuse/growth must wait for the previous upload, even when no
    # caller has synchronized that stream yet.
    calc = _DistributedMultiplier((0, destination), transfer="host")
    streams = []
    outputs = []
    try:
        with cp.cuda.Device(destination):
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        for i, count in enumerate([(1 << 20) + 17, 33, (1 << 21) + 5, 1 << 20]):
            with cp.cuda.Device(0):
                source = cp.full(count, i + 7, dtype=cp.uint64)
                cp.cuda.get_current_stream().synchronize()
            with cp.cuda.Device(destination):
                target = cp.empty(count, dtype=cp.uint64)
                calc._copy(target, source, streams[i % 2])
            outputs.append((target, i + 7))
        with cp.cuda.Device(destination):
            for stream in streams:
                stream.synchronize()
            for target, expected in outputs:
                assert bool(cp.all(target == expected))
        assert len(calc._host_buffers) == 1
        assert calc.host_copies == len(outputs)
    finally:
        with cp.cuda.Device(destination):
            for stream in streams:
                stream.synchronize()
        calc.close()
