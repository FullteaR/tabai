"""CPU-only regression tests for benchmark inputs, errors, isolation and CLI."""
import builtins
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks import common, runner


def config(**overrides):
    values = dict(backends=['python'], gpu_mode='single', devices=[0],
                  operations=['add'], bits=[32], pow_exponents=[2, 3],
                  warmup=0, repeat=1, seed=42, min_bits=64_000_000,
                  transfer='auto', verify=True, timeout=5.0)
    values.update(overrides)
    return values


def test_case_inputs_independent_of_backend_order_and_other_cases():
    case = dict(operation='mul', bits=257, exponent=None)
    first = common.make_inputs(case, 42)
    common.make_inputs(dict(operation='pow', bits=33, exponent=2), 42)
    second = common.make_inputs(case, 42)
    assert first == second
    assert common.input_digest(first) == common.input_digest(second)
    assert common.make_inputs(case, 43) != first


@pytest.mark.parametrize('bits', [2, 3, 32, 33, 128])
def test_subtraction_inputs_have_exact_width(bits):
    for seed in range(30):
        a, b = common.make_inputs(dict(operation='sub', bits=bits), seed)
        assert a > b and a.bit_length() == b.bit_length() == bits


def test_square_preserves_operand_identity():
    class Operand:
        def __mul__(self, other):
            assert self is other
            return 49
    assert common.calculate('square', (Operand(),)) == 49


def test_timing_syncs_warmup_and_measures_only_operation():
    events = []
    ticks = iter([0., 100., 200., 204., 300., 306.])
    result = common.measure(lambda: events.append('run') or 42,
                            sync=lambda: events.append('sync'),
                            verify=lambda value: events.append(('verify', value)),
                            warmup=1, repeat=2, clock=lambda: next(ticks))
    assert result == {'median_seconds': 5., 'samples_seconds': [4., 6.]}
    assert events == ['sync', 'run', 'sync', ('verify', 42)] * 3


def test_cpu_benchmark_does_not_import_cuda(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(('cupy', 'tabai_gpu')):
            pytest.fail('CPU benchmark touched CUDA')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    record = common.run_case(common.PythonBackend(), dict(operation='pow', bits=64, exponent=3),
                             config(), lambda phase: None)
    assert record['status'] == 'ok' and record['verified']


@pytest.mark.parametrize('kind, phase', [('conversion', 'preparation'), ('sync', 'measurement')])
def test_conversion_and_async_errors_are_not_timeouts(kind, phase):
    class BrokenBackend(common.PythonBackend):
        def from_int(self, value):
            if kind == 'conversion':
                raise MemoryError('cannot allocate operands')
            return value
        def sync(self):
            raise RuntimeError('asynchronous kernel failed')
    record = common.run_case(BrokenBackend(), dict(operation='add', bits=32), config(), lambda p: None)
    assert record['status'] == ('oom' if kind == 'conversion' else 'error')
    assert record['phase'] == phase
    assert 'median_seconds' not in record


def test_reference_mismatch_is_reported():
    class Incorrect(common.PythonBackend):
        def to_int(self, result):
            return -1
    record = common.run_case(Incorrect(), dict(operation='add', bits=32), config(), lambda p: None)
    assert record['status'] == 'mismatch' and record['phase'] == 'verification'


def _stalled_worker(connection):
    connection.send({'kind': 'ready'})
    connection.recv()
    connection.send({'kind': 'phase', 'phase': 'measurement'})
    time.sleep(60)


def test_timeout_kills_worker_and_releases_process():
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    worker = runner.Worker({'backend': 'python'}, config(timeout=0.05))
    worker.connection = parent
    worker.process = context.Process(target=_stalled_worker, args=(child,))
    worker.process.start()
    child.close()
    pid = worker.process.pid
    try:
        assert runner.receive(parent, 10, 'initialization')['kind'] == 'ready'
        start = time.monotonic()
        record = worker.run(dict(operation='add', bits=32, exponent=None))
        assert record['status'] == 'timeout' and record['phase'] == 'measurement'
        assert time.monotonic() - start < 5
        assert worker.process is None and worker.connection is None
        assert pid not in {p.pid for p in mp.active_children()}
    finally:
        worker.stop()


def test_failure_skips_only_larger_sizes_for_same_exponent(monkeypatch):
    calls = []
    class FakeWorker:
        def __init__(self, spec, cfg):
            self.process = None
            self.metadata = {}
        def start(self):
            self.process = True
        def run(self, case):
            calls.append((case['bits'], case['exponent']))
            if case['bits'] == 32 and case['exponent'] == 2:
                self.process = None
                return dict(case, status='oom', error='simulated')
            return dict(case, status='ok', median_seconds=0.001)
        def stop(self, graceful=False):
            pass
    monkeypatch.setattr(runner, 'Worker', FakeWorker)
    report = {'reports': []}
    runner.run_suite(config(operations=['pow'], bits=[32, 64]), report)
    assert calls == [(32, 2), (32, 3), (64, 3)]
    assert [r['status'] for r in report['reports'][0]['rows']] == ['oom', 'ok', 'skipped', 'ok']


@pytest.mark.parametrize('args', [
    ['--repeat', '0'], ['--warmup', '-1'], ['--timeout', 'nan'], ['--timeout', 'inf'],
    ['--timeout', '0'], ['--bits', '1'], ['--pow-exponents', '-1'],
    ['--backends', 'tabai', '--gpu-mode', 'both', '--devices', '0'],
    ['--backends', 'tabai', '--gpu-mode', 'single', '--devices', '0', '1'],
    ['--backends', 'tabai', '--gpu-mode', 'multi', '--devices', '0', '0'],
])
def test_invalid_cli_options_fail_early(args):
    with pytest.raises(SystemExit) as error:
        runner.parse_args(args)
    assert error.value.code == 2


def test_cpu_cli_and_module_entry_use_identical_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    files = [tmp_path / 'direct.json', tmp_path / 'module.json']
    commands = [[sys.executable, str(root / 'benchmarks/benchmark.py')],
                [sys.executable, '-m', 'benchmarks.benchmark']]
    for command, output in zip(commands, files):
        done = subprocess.run(command + ['--backends', 'python', '--bits', '32', '64',
                              '--pow-exponents', '0', '3', '--warmup', '0', '--repeat', '1',
                              '--output', str(output)], cwd=root, text=True, capture_output=True, timeout=20)
        assert done.returncode == 0, done.stderr
    reports = [json.loads(p.read_text()) for p in files]
    first, second = [r['reports'][0]['rows'] for r in reports]
    assert len(first) == len(second) == 16
    assert [row['input_sha256'] for row in first] == [row['input_sha256'] for row in second]
    assert all(row['status'] == 'ok' and row['verified'] for row in first + second)
