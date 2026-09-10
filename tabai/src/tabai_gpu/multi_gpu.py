"""Sharded Goldilocks NTT for one large multiplication across CUDA devices."""
from contextlib import contextmanager
from operator import index

import cupy as cp
import numpy as np

from .core import (
    GPUBigInt, _BLOCK, _GL_P, _GL_PRIMITIVE_ROOT, _GL_PREAMBLE,
    _distributed_multiplier, _ntt_pointwise_scale_kernel,
)

# Pipeline host staging above one chunk; buffers stay pinned until upload.
_HOST_CHUNK_BYTES = 2 * 1024 * 1024

# Each rank owns a contiguous n/ranks-element interval. Cross-rank butterflies
# use a snapshot of the partner interval; no kernel reads a remote pointer.
_twiddles_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void shard_twiddles(unsigned long long* w, unsigned long long base,
                    unsigned long long offset, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long e = offset + (unsigned long long)i;
    unsigned long long v = 1ULL;
    while (e) {
        if (e & 1ULL) v = gl_mulmod(v, base);
        base = gl_mulmod(base, base);
        e >>= 1;
    }
    w[i] = v;
}
''', 'shard_twiddles')

_cross_stage_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void shard_stage(unsigned long long* a, const unsigned long long* partner,
                 const unsigned long long* w, int n, int upper, int inverse)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long u = upper ? partner[i] : a[i];
    unsigned long long v = upper ? a[i] : partner[i];
    if (inverse) {
        unsigned long long t = gl_mulmod(v, w[i]);
        a[i] = upper ? gl_submod(u, t) : gl_addmod(u, t);
    } else {
        a[i] = upper ? gl_mulmod(gl_submod(u, v), w[i]) : gl_addmod(u, v);
    }
}
''', 'shard_stage')


# With two ranks, finish the last inverse butterfly directly into the primary
# carry buffer. Only rank 1 -> rank 0 is transferred; there is no return trip
# to rank 1 followed by a second gather of those same coefficients.
_gather_inverse_kernel = cp.RawKernel(_GL_PREAMBLE + r'''
extern "C" __global__
void gather_inverse(const unsigned long long* lower,
                    const unsigned long long* upper,
                    const unsigned long long* w,
                    unsigned long long* coefficients, int half, int n_conv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    unsigned long long t = gl_mulmod(upper[i], w[i]);
    coefficients[i] = gl_addmod(lower[i], t);
    if (i + half < n_conv) coefficients[i + half] = gl_submod(lower[i], t);
}
''', 'gather_inverse')


def _layout(la, lb, ranks):
    """Lengths only: validate before allocating large GPU buffers."""
    if la < 1 or lb < 1 or ranks < 1 or ranks & (ranks - 1):
        raise ValueError("nonempty operands and a power-of-two rank count required")
    n_a, n_b = 2 * la, 2 * lb
    if min(n_a, n_b) * 65535**2 >= 1 << 63:
        raise ValueError("convolution coefficients exceed the int64 carry bound")
    n_conv = n_a + n_b - 1
    n = max(ranks, 1 << (n_conv - 1).bit_length())
    if n > 1 << 32:
        raise ValueError("NTT length exceeds the Goldilocks limit of 2**32")
    # Existing carry kernels and local NTT kernels take signed int lengths.
    if n_conv + 1 >= 1 << 31 or n // ranks >= 1 << 31:
        raise ValueError("operand size exceeds the current CUDA kernel index range")
    return n_a, n_b, n_conv, n, n // ranks


class _Rank:
    def __init__(self, device_id):
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            with self.stream:
                self.engine = GPUBigInt(max_bits=0)
        self.capacity = 0
        self.tables = {}

    def ensure(self, size):
        if size > self.capacity:
            self.a = cp.empty(size, dtype=cp.uint64)
            self.b = cp.empty(size, dtype=cp.uint64)
            self.exchange = cp.empty(size, dtype=cp.uint64)
            self.capacity = size

    def twiddles(self, size, half, offset, inverse):
        key = (size, half, offset, inverse)
        if key not in self.tables:
            root = pow(_GL_PRIMITIVE_ROOT, (_GL_P - 1) // (2 * half), _GL_P)
            if inverse:
                root = pow(root, _GL_P - 2, _GL_P)
            table = cp.empty(size, dtype=cp.uint64)
            _twiddles_kernel(
                ((size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                (table, np.uint64(root), np.uint64(offset), size))
            self.tables[key] = table
        return self.tables[key]


class _DistributedMultiplier:
    """Private executor; repeated devices are permitted only for rank tests.

    The public context validates distinct physical devices. Tests run the real
    CUDA algorithm with 2/4 logical ranks on one card, which does not validate
    peer transfers or simultaneous execution on multiple physical cards.
    """
    def __init__(self, devices, min_bits=64_000_000, transfer="auto"):
        self.devices = tuple(devices)
        self.min_bits = min_bits
        self.transfer = transfer
        self.ranks = [_Rank(d) for d in devices]
        self.multiplications = 0
        self.peer_copies = 0
        self.host_copies = 0
        self.local_copies = 0
        self._peer_access = {}
        self._host_buffers = {}
        self._host_uploads = {}
        self._download_streams = {}

    def synchronize(self):
        for device_id, stream in self._download_streams.items():
            with cp.cuda.Device(device_id):
                stream.synchronize()
        for rank in self.ranks:
            with cp.cuda.Device(rank.device_id):
                rank.stream.synchronize()

    def close(self):
        self.synchronize()
        self.ranks.clear()
        self._host_buffers.clear()
        self._host_uploads.clear()
        self._download_streams.clear()

    def _copy(self, dst, src, stream):
        """Sources must be complete; destination work is ordered on stream.

        Host staging is explicit when P2P is unavailable. Keep the staging
        array alive until the upload finishes. No topology errors are hidden.
        """
        if dst.nbytes != src.nbytes or dst.dtype != src.dtype:
            raise ValueError("transfer shape/dtype mismatch")
        if not dst.nbytes:
            return
        src_id, dst_id = src.device.id, dst.device.id
        with cp.cuda.Device(dst_id):
            if self.transfer != "host":
                if src_id == dst_id:
                    dst.data.copy_from_device_async(src.data, src.nbytes, stream)
                    self.local_copies += 1
                    return
                pair = (dst_id, src_id)
                if pair not in self._peer_access:
                    self._peer_access[pair] = cp.cuda.runtime.deviceCanAccessPeer(*pair)
                if self._peer_access[pair]:
                    dst.data.copy_from_device_async(src.data, src.nbytes, stream)
                    self.peer_copies += 1
                    return
        # Reuse pinned storage only after its previous upload completes. Wait
        # for that upload's event, not all later work on the destination stream:
        # rank NTT kernels can overlap preparation of the next operand.
        key = (src_id, dst_id, src.dtype.str)
        previous = self._host_uploads.get(key)
        if previous is not None:
            with cp.cuda.Device(dst_id):
                previous.synchronize()
        host = self._host_buffers.get(key)
        if host is None or host.nbytes < src.nbytes:
            memory = cp.cuda.alloc_pinned_memory(src.nbytes)
            host = np.frombuffer(memory, dtype=src.dtype, count=src.size)
            self._host_buffers[key] = host
        host = host[:src.size].reshape(src.shape)
        if (src.nbytes > _HOST_CHUNK_BYTES and src.flags.c_contiguous
                and dst.flags.c_contiguous):
            # All sources are already complete at the callers' barriers. A
            # dedicated source stream downloads chunks ahead of the uploads,
            # allowing D2H and H2D on different cards to run concurrently.
            with cp.cuda.Device(src_id):
                download = self._download_streams.get(src_id)
                if download is None:
                    download = self._download_streams[src_id] = cp.cuda.Stream(non_blocking=True)
                source = src.reshape(-1)
                flat_host = host.reshape(-1)
                step = _HOST_CHUNK_BYTES // src.dtype.itemsize
                pending = []
                for start in range(0, src.size, step):
                    stop = min(start + step, src.size)
                    source[start:stop].get(out=flat_host[start:stop], stream=download, blocking=False)
                    ready = cp.cuda.Event(disable_timing=True)
                    ready.record(download)
                    pending.append((start, stop, ready))
            with cp.cuda.Device(dst_id):
                target = dst.reshape(-1)
            for start, stop, ready in pending:
                with cp.cuda.Device(src_id):
                    ready.synchronize()  # this chunk is now readable by the CPU/GPU
                with cp.cuda.Device(dst_id):
                    target[start:stop].set(flat_host[start:stop], stream=stream)
        else:
            with cp.cuda.Device(src_id):
                src.get(out=host, blocking=True)
            with cp.cuda.Device(dst_id):
                dst.set(host, stream=stream)
        with cp.cuda.Device(dst_id):
            uploaded = cp.cuda.Event(disable_timing=True)
            uploaded.record(stream)
            self._host_uploads[key] = uploaded
        self.host_copies += 1

    def _scatter(self, source, name, size):
        for i, rank in enumerate(self.ranks):
            start = i * size
            count = max(0, min(size, len(source) - start))
            with cp.cuda.Device(rank.device_id), rank.stream:
                target = getattr(rank, name)[:size]
                if count:
                    # Transfer uint16 input without replicating the full operand.
                    staging = rank.exchange.view(cp.uint16)[:count]
                    self._copy(staging, source[start:start + count], rank.stream)
                    target[:count] = staging
                target[count:] = 0

    def _prepare_forward(self, source, name, n, size):
        if len(self.ranks) != 2 or len(source) > size:
            self._scatter(source, name, size)
            self._forward(name, n, size)
            return
        # The upper input half is entirely zero. The first DIF stage is just
        # lower=u, upper=u*w, so send uint16 input to both cards instead of
        # exchanging uint64 intervals (including one all-zero interval).
        for i, rank in enumerate(self.ranks):
            with cp.cuda.Device(rank.device_id), rank.stream:
                target = getattr(rank, name)[:size]
                staging = rank.exchange.view(cp.uint16)[:len(source)]
                self._copy(staging, source, rank.stream)
                target[:len(source)] = staging
                target[len(source):] = 0
                if i:
                    w = rank.twiddles(size, size, 0, inverse=False)
                    _ntt_pointwise_scale_kernel(
                        ((size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                        (target, w, np.uint64(1), size))
                w, _, _ = rank.engine._get_ntt_tables(size)
                rank.engine._ntt_forward(target, size, w)

    def _inverse_two(self, engine, n_conv, size, caller_stream):
        for rank in self.ranks:
            with cp.cuda.Device(rank.device_id), rank.stream:
                _, w, _ = rank.engine._get_ntt_tables(size)
                rank.engine._ntt_inverse(rank.a, size, w)
        self.synchronize()
        lower, upper = self.ranks
        with cp.cuda.Device(lower.device_id), lower.stream:
            self._copy(lower.exchange[:size], upper.a[:size], lower.stream)
            w = lower.twiddles(size, size, 0, inverse=True)
        lower.stream.synchronize()
        engine._ensure_carry_capacity(n_conv + 1)
        ping = engine._carry_ping[:n_conv + 1]
        _gather_inverse_kernel(
            ((size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
            (lower.a, lower.exchange, w, ping, size, n_conv))
        caller_stream.synchronize()  # release rank buffers before scope exit/reuse
        return ping

    def _cross_stage(self, name, size, half, inverse):
        distance = half // size
        self.synchronize()
        # Finish ALL partner snapshots before overwriting ANY source interval.
        for i, rank in enumerate(self.ranks):
            partner = self.ranks[i ^ distance]
            with cp.cuda.Device(rank.device_id), rank.stream:
                self._copy(rank.exchange[:size], getattr(partner, name)[:size], rank.stream)
        self.synchronize()
        for i, rank in enumerate(self.ranks):
            upper = bool(i & distance)
            offset = ((i & ~distance) * size) % half
            with cp.cuda.Device(rank.device_id), rank.stream:
                # Forward lower ranks do not read w; the exchange array is a
                # valid dummy pointer. Inverse ranks both need inverse twiddles.
                w = rank.twiddles(size, half, offset, inverse) if upper or inverse else rank.exchange
                _cross_stage_kernel(
                    ((size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                    (getattr(rank, name), rank.exchange, w, size, int(upper), int(inverse)))

    def _forward(self, name, n, size):
        half = n // 2
        while half >= size:
            self._cross_stage(name, size, half, inverse=False)
            half //= 2
        for rank in self.ranks:
            with cp.cuda.Device(rank.device_id), rank.stream:
                w, _, _ = rank.engine._get_ntt_tables(size)
                rank.engine._ntt_forward(getattr(rank, name), size, w)

    def _inverse(self, name, n, size):
        for rank in self.ranks:
            with cp.cuda.Device(rank.device_id), rank.stream:
                _, w, _ = rank.engine._get_ntt_tables(size)
                rank.engine._ntt_inverse(getattr(rank, name), size, w)
        half = size
        while half < n:
            self._cross_stage(name, size, half, inverse=True)
            half *= 2

    def mul(self, engine, a, b):
        primary = self.devices[0]
        if (cp.cuda.runtime.getDevice() != primary or engine.device_id != primary
                or a.device.id != primary or b.device.id != primary):
            raise ValueError("operands, engine and current device must use devices[0]")
        n_a, n_b, n_conv, n, size = _layout(len(a), len(b), len(self.ranks))
        square = a is b
        a_ch = cp.ascontiguousarray(a).view(cp.uint16)
        b_ch = a_ch if square else cp.ascontiguousarray(b).view(cp.uint16)
        caller_stream = cp.cuda.get_current_stream()
        caller_stream.synchronize()  # inputs and prior uses of the carry buffer
        try:
            for rank in self.ranks:
                with cp.cuda.Device(rank.device_id), rank.stream:
                    rank.ensure(size)
            self._prepare_forward(a_ch, "a", n, size)
            if not square:
                self._prepare_forward(b_ch, "b", n, size)
            inv_n = np.uint64(pow(n, _GL_P - 2, _GL_P))
            for rank in self.ranks:
                with cp.cuda.Device(rank.device_id), rank.stream:
                    _ntt_pointwise_scale_kernel(
                        ((size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                        (rank.a, rank.a if square else rank.b, inv_n, size))
            if len(self.ranks) == 2:
                ping = self._inverse_two(engine, n_conv, size, caller_stream)
            else:
                self._inverse("a", n, size)
                self.synchronize()

                # Gather only the linear convolution, discarding zero padding.
                # Coefficients below 2**63 have the same int64/uint64 bit pattern.
                engine._ensure_carry_capacity(n_conv + 1)
                ping = engine._carry_ping[:n_conv + 1]
                for i, rank in enumerate(self.ranks):
                    start = i * size
                    count = max(0, min(size, n_conv - start))
                    if count:
                        self._copy(ping[start:start + count].view(cp.uint64),
                                   rank.a[:count], caller_stream)
                caller_stream.synchronize()
            ping[n_conv] = 0
            max_bits = 32 + min(n_a, n_b).bit_length()
            ping = engine._resolve_carries(n_conv + 1, 16, max_bits)
            result = engine._recombine_chunks(ping, n_conv + 1, 16, len(a) + len(b))
            self.multiplications += 1
            return result
        except BaseException:
            # Drain pending work before an exception unwinds owners of buffers.
            self.synchronize()
            caller_stream.synchronize()
            raise


@contextmanager
def multi_gpu(devices=(0, 1), *, min_bits=64_000_000, transfer="auto"):
    """Distribute large NTT multiplications in this scope over 1/2/4/... GPUs.

    The 64 Mbit default avoids the small-input transfer overhead observed on
    two RTX 3090s using host staging; it is configurable for other topologies.
    Create and operate on TabaiInt values on devices[0], the current device.
    Schoolbook multiplication and NTT operands below min_bits stay on that GPU.
    Multiplications inside powers and Newton division use the same dispatch.
    transfer='auto' uses P2P when available, otherwise CPU staging; 'host' forces
    staging for validation. Device IDs are CUDA-visible ordinals.
    """
    devices = tuple(index(d) for d in devices)
    min_bits = index(min_bits)
    if not devices or len(devices) & (len(devices) - 1):
        raise ValueError("select a power-of-two number of devices (1, 2, 4, ...)")
    if len(set(devices)) != len(devices):
        raise ValueError("device IDs must be distinct physical GPUs")
    count = cp.cuda.runtime.getDeviceCount()
    if any(d < 0 or d >= count for d in devices):
        raise ValueError(f"device IDs must be in [0, {count})")
    if cp.cuda.runtime.getDevice() != devices[0]:
        raise ValueError("enter cp.cuda.Device(devices[0]) before multi_gpu")
    if min_bits < 0 or transfer not in ("auto", "host"):
        raise ValueError("min_bits must be nonnegative; transfer must be 'auto' or 'host'")
    executor = _DistributedMultiplier(devices, min_bits, transfer)
    token = _distributed_multiplier.set(executor)
    try:
        yield executor
    finally:
        _distributed_multiplier.reset(token)
        executor.close()
