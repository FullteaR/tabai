"""Deterministic inputs, backend adapters and timing (no eager CUDA imports)."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import operator
import random
import statistics
import time


BIT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
             1_000_000_000, 10_000_000_000]
DIV_BIT_SIZES = BIT_SIZES[:-1]
POW_BASE_BITS = BIT_SIZES[:5]
POW_EXPONENTS = [2, 3, 10, 20]
OPERATIONS = ("add", "sub", "mul", "square", "div", "mod", "pow")


def random_int(bits, rng):
    """Generate an exact-width positive integer, including widths > 2**31."""
    value = 0
    remaining = bits
    while remaining:
        width = min(remaining, 1 << 29)
        value = (value << width) | rng.getrandbits(width)
        remaining -= width
    return value | (1 << (bits - 1))


def make_inputs(case, seed):
    # Seed each case independently: backend ordering, omitted operations and
    # skipped rows cannot change another backend's input values.
    op, bits, exponent = case['operation'], case['bits'], case.get('exponent')
    rng = random.Random(f"tabai:{seed}:{op}:{bits}:{exponent}")
    a = random_int(bits, rng)
    if op == 'square':
        return (a,)
    if op == 'pow':
        return a, exponent
    b = random_int(max(1, bits // 2) if op in ('div', 'mod') else bits, rng)
    if op == 'sub':
        a, b = max(a, b), min(a, b)
        if a == b:
            if b > 1 << (bits - 1):
                b -= 1
            else:
                a += 1  # bits >= 2; both operands retain the requested width
    return a, b


def input_digest(values):
    digest = hashlib.sha256()
    for value in values:
        data = value.to_bytes(max(1, (value.bit_length() + 7) // 8), 'little')
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return digest.hexdigest()


def square(value):
    return value * value  # preserve identity for TabaiInt's square path


OP_FUNCTIONS = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
                'div': operator.floordiv, 'mod': operator.mod, 'pow': operator.pow,
                'square': square}


def calculate(operation, values):
    return OP_FUNCTIONS[operation](*values)


def format_time(seconds):
    if seconds < 1e-3:
        return f'{seconds * 1e6:.3f} us'
    if seconds < 1:
        return f'{seconds * 1e3:.3f} ms'
    return f'{seconds:.3f} s'


class ResultMismatch(Exception):
    pass


def measure(fn, *, sync=lambda: None, verify=None, notify=lambda phase: None,
            warmup=2, repeat=5, clock=time.perf_counter):
    if warmup < 0 or repeat < 1:
        raise ValueError('warmup must be >= 0 and repeat >= 1')
    samples = []
    for iteration in range(warmup + repeat):
        notify('warmup' if iteration < warmup else 'measurement')
        sync()  # includes completion of input conversion, also before warmup
        start = clock()
        result = fn()
        sync()  # asynchronous CUDA failures propagate to the worker's handler
        elapsed = clock() - start
        if verify is not None:
            notify('verification')
            verify(result)
        if iteration >= warmup:
            samples.append(elapsed)
        # Release old output BEFORE the next timed allocation, not in result=fn().
        del result
    return {'median_seconds': statistics.median(samples), 'samples_seconds': samples}


class PythonBackend:
    def context(self):
        return nullcontext()

    def from_int(self, value):
        return value

    def to_int(self, value):
        return int(value)

    def sync(self):
        pass  # CPU timing must never initialize or synchronize CUDA

    def counters(self):
        return {'distributed_multiplications': 0, 'copies': {'peer': 0, 'host': 0, 'local': 0}}

    def memory(self):
        return {}

    def metadata(self):
        return {}


class GmpBackend(PythonBackend):
    def __init__(self):
        import gmpy2
        self.gmpy2 = gmpy2

    def from_int(self, value):
        return self.gmpy2.mpz(value)

    def metadata(self):
        return {'gmpy2_version': self.gmpy2.version()}


class TabaiBackend(PythonBackend):
    def __init__(self, spec, config):
        import cupy as cp
        from tabai_gpu import TabaiInt, multi_gpu
        self.cp, self.TabaiInt, self.multi_gpu = cp, TabaiInt, multi_gpu
        self.devices = spec['devices']
        self.mode = spec['gpu_mode']
        self.config = config
        self.executor = None
        count = cp.cuda.runtime.getDeviceCount()
        if any(d < 0 or d >= count for d in self.devices):
            raise ValueError(f"requested devices {self.devices}; only {count} CUDA device(s) visible")

    @contextmanager
    def context(self):
        with self.cp.cuda.Device(self.devices[0]):
            scope = (self.multi_gpu(self.devices, min_bits=self.config['min_bits'],
                                    transfer=self.config['transfer'])
                     if self.mode == 'multi' else nullcontext())
            with scope as self.executor:
                yield

    def from_int(self, value):
        return self.TabaiInt(value)

    def to_int(self, value):
        return value.to_cpu()

    def sync(self):
        if self.executor is not None:
            self.executor.synchronize()
        self.cp.cuda.get_current_stream().synchronize()

    def counters(self):
        if self.executor is None:
            return super().counters()
        return {'distributed_multiplications': self.executor.multiplications,
                'copies': {k: getattr(self.executor, f'{k}_copies') for k in ('peer', 'host', 'local')}}

    def memory(self):
        reserved = {}
        for device in self.devices:
            with self.cp.cuda.Device(device):
                reserved[str(device)] = self.cp.get_default_memory_pool().total_bytes()
        return reserved

    def metadata(self):
        hardware = []
        for device in self.devices:
            props = self.cp.cuda.runtime.getDeviceProperties(device)
            name = props['name']
            hardware.append({'id': device, 'name': name.decode() if isinstance(name, bytes) else name,
                             'total_memory_bytes': props['totalGlobalMem']})
        return {'hardware': hardware, 'cupy_version': self.cp.__version__,
                'cuda_runtime': self.cp.cuda.runtime.runtimeGetVersion(),
                'cuda_driver': self.cp.cuda.runtime.driverGetVersion(),
                'peer_access': {f'{dst}<-{src}': bool(self.cp.cuda.runtime.deviceCanAccessPeer(dst, src))
                                for dst in self.devices for src in self.devices if dst != src}}


def create_backend(spec, config):
    if spec['backend'] == 'python':
        return PythonBackend()
    if spec['backend'] == 'gmpy2':
        return GmpBackend()
    return TabaiBackend(spec, config)


def run_case(backend, case, config, notify):
    phase = 'preparation'
    record = dict(case, status='ok')

    def progress(value):
        nonlocal phase
        phase = value
        notify(value)

    try:
        progress('preparation')
        raw = make_inputs(case, config['seed'])
        record['input_sha256'] = input_digest(raw)
        args = tuple(backend.from_int(v) for v in raw)
        operation = OP_FUNCTIONS[case['operation']]
        verify = None
        if config['verify']:
            progress('reference')
            try:
                oracle = GmpBackend()
            except ImportError:
                oracle = PythonBackend()
            expected = operation(*(oracle.from_int(v) for v in raw))

            def verify(result):
                if backend.to_int(result) != expected:
                    raise ResultMismatch('result differs from the CPU integer reference')

        before = backend.counters()
        record.update(measure(lambda: operation(*args), sync=backend.sync,
                              verify=verify, notify=progress, warmup=config['warmup'], repeat=config['repeat']))
        after = backend.counters()
        progress('reporting')
        record.update(verified=config['verify'],
                      distributed_multiplications=after['distributed_multiplications'] - before['distributed_multiplications'],
                      copies={k: after['copies'][k] - before['copies'][k] for k in after['copies']},
                      cupy_pool_reserved_bytes=backend.memory())
    except Exception as error:
        # GPU OutOfMemoryError subclasses MemoryError. Unexpected exceptions
        # and incorrect results remain errors instead of being called timeouts.
        status = 'oom' if isinstance(error, MemoryError) else 'error'
        if isinstance(error, ResultMismatch):
            status = 'mismatch'
        record.update(status=status, phase=phase, error_type=type(error).__name__, error=str(error))
    return record
