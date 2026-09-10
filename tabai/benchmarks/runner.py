"""Shared CLI and isolated backend workers for all benchmark entry points."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys

if __package__:
    from .common import BIT_SIZES, DIV_BIT_SIZES, POW_BASE_BITS, POW_EXPONENTS, OPERATIONS, create_backend, run_case, format_time
else:
    from common import BIT_SIZES, DIV_BIT_SIZES, POW_BASE_BITS, POW_EXPONENTS, OPERATIONS, create_backend, run_case, format_time


def parse_args(argv=None, *, operations=OPERATIONS, gpu_mode='single', backends=None,
               bits=None, exponents=POW_EXPONENTS):
    parser = argparse.ArgumentParser(description='Compare deterministic integer workloads on CPU, one GPU, or multiple GPUs.')
    parser.add_argument('--gpu-mode', '--mode', choices=['single', 'multi', 'both'], default=gpu_mode)
    parser.add_argument('--devices', type=int, nargs='+', help='CUDA-visible IDs; single uses one ID, multi/both use 2/4/... IDs')
    parser.add_argument('--backends', choices=['tabai', 'gmpy2', 'python'], nargs='+', default=backends)
    parser.add_argument('--operations', choices=OPERATIONS, nargs='+', default=list(operations))
    parser.add_argument('--bits', type=int, nargs='+', default=bits, help='Override the input bit sizes for all selected operations')
    parser.add_argument('--pow-exponents', type=int, nargs='+', default=list(exponents))
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=10.0, help='Seconds per preparation, reference, warmup, measurement or verification stage; kills stalled worker')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-bits', type=int, default=64_000_000)
    parser.add_argument('--transfer', choices=['auto', 'host'], default='auto')
    parser.add_argument('--no-verify', dest='verify', action='store_false', help='Skip CPU-reference calculation and result checks')
    parser.add_argument('--output', type=Path, help='Save JSON, including failures and partial results on interruption')
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1 or args.min_bits < 0:
        parser.error('warmup/min-bits must be >= 0; repeat must be >= 1')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive and finite')
    if args.bits is not None and min(args.bits) < 2:
        parser.error('bits must be >= 2')
    if min(args.pow_exponents) < 0:
        parser.error('pow-exponents must be nonnegative')
    if args.backends is None:
        args.backends = []
        if args.gpu_mode != 'single' or (importlib.util.find_spec('tabai_gpu') is not None and importlib.util.find_spec('cupy') is not None):
            args.backends.append('tabai')
        if importlib.util.find_spec('gmpy2') is not None:
            args.backends.append('gmpy2')
        args.backends.append('python')
    args.backends = list(dict.fromkeys(args.backends))
    args.operations = list(dict.fromkeys(args.operations))
    args.pow_exponents = sorted(set(args.pow_exponents))
    if args.bits is not None:
        args.bits = sorted(set(args.bits))
    if args.devices is None:
        args.devices = [0] if args.gpu_mode == 'single' else [0, 1]
    if 'tabai' in args.backends:
        if min(args.devices) < 0 or len(set(args.devices)) != len(args.devices):
            parser.error('devices must be distinct nonnegative CUDA IDs')
        count = len(args.devices)
        if args.gpu_mode == 'single' and count != 1:
            parser.error('single mode requires exactly one device')
        if args.gpu_mode != 'single' and (count < 2 or count & (count - 1)):
            parser.error('multi/both mode requires 2, 4, ... distinct GPUs')
    return args


def cases_for(config):
    cases = []
    for operation in config['operations']:
        sizes = config['bits']
        if sizes is None:
            sizes = POW_BASE_BITS if operation == 'pow' else DIV_BIT_SIZES if operation in ('div', 'mod') else BIT_SIZES
        for bits in sizes:
            for exponent in config['pow_exponents'] if operation == 'pow' else [None]:
                cases.append({'operation': operation, 'bits': bits, 'exponent': exponent})
    return cases


def specs_for(config):
    specs = []
    for backend in config['backends']:
        if backend == 'tabai':
            modes = ['single', 'multi'] if config['gpu_mode'] == 'both' else [config['gpu_mode']]
            for mode in modes:
                devices = config['devices'][:1] if mode == 'single' else config['devices']
                specs.append({'name': f'tabai-{mode}', 'backend': backend,
                              'gpu_mode': mode, 'devices': devices})
        else:
            specs.append({'name': backend, 'backend': backend, 'gpu_mode': None, 'devices': []})
    return specs


def worker_main(connection, spec, config):
    try:
        backend = create_backend(spec, config)
        with backend.context():
            connection.send({'kind': 'ready', 'metadata': backend.metadata()})
            while True:
                case = connection.recv()
                if case is None:
                    break
                record = run_case(backend, case, config,
                                  lambda phase: connection.send({'kind': 'phase', 'phase': phase}))
                connection.send({'kind': 'result', 'record': record})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send({'kind': 'fatal', 'error_type': type(error).__name__, 'error': str(error)})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


def receive(connection, timeout, initial_phase):
    """Wait one bounded stage at a time; status messages reset the deadline."""
    phase = initial_phase
    while True:
        if not connection.poll(timeout):
            return {'kind': 'failed', 'status': 'timeout', 'phase': phase,
                    'error': f'{phase} exceeded {timeout:g} seconds'}
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return {'kind': 'failed', 'status': 'error', 'phase': phase,
                    'error': 'worker exited without returning a result'}
        if message['kind'] == 'phase':
            phase = message['phase']
            continue
        if message['kind'] == 'fatal':
            return dict(message, kind='failed', status='error', phase=phase)
        return message


class Worker:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.process = self.connection = None
        self.metadata = {}

    def start(self):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child, self.spec, self.config), daemon=True)
        try:
            self.process.start()
        finally:
            child.close()
        message = receive(self.connection, max(30.0, self.config['timeout']), 'initialization')
        if message['kind'] != 'ready':
            self.stop()
            return message
        self.metadata = message['metadata']
        return None

    def run(self, case):
        try:
            self.connection.send(case)
            message = receive(self.connection, self.config['timeout'], 'preparation')
        except (BrokenPipeError, EOFError, OSError) as error:
            message = {'kind': 'failed', 'status': 'error', 'phase': 'preparation', 'error': str(error)}
        if message['kind'] == 'result':
            record = message['record']
        else:
            record = dict(case, **{k: v for k, v in message.items() if k != 'kind'})
        if record['status'] != 'ok':
            # CUDA may be in an error state, or stale large buffers may fill VRAM.
            # Kill/drain this process before any other mode starts or is retried.
            self.stop()
        return record

    def stop(self, graceful=False):
        if self.process is not None:
            if graceful and self.process.is_alive():
                try:
                    self.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def run_suite(config, report):
    cases = cases_for(config)
    for spec in specs_for(config):
        print(f"Measuring {spec['name']} devices={spec['devices']} ...", file=sys.stderr, flush=True)
        backend_report = dict(spec, rows=[])
        report['reports'].append(backend_report)
        worker = Worker(spec, config)
        limits = {}
        try:
            for case in cases:
                key = (case['operation'], case['exponent'])
                if key in limits and case['bits'] > limits[key]:
                    record = dict(case, status='skipped', error='larger input after timeout/OOM')
                else:
                    if worker.process is None:
                        failed = worker.start()
                        if failed is not None:
                            failure = {k: v for k, v in failed.items() if k != 'kind'}
                            backend_report['initialization_error'] = failure
                            # Initialization failure applies to every remaining case.
                            for remaining in cases[len(backend_report['rows']):]:
                                backend_report['rows'].append(dict(remaining, **failure))
                            print(f"  initialization failed: {failure['error']}", file=sys.stderr, flush=True)
                            break
                        backend_report.update(worker.metadata)
                    record = worker.run(case)
                    if record['status'] in ('timeout', 'oom'):
                        limits[key] = case['bits']
                backend_report['rows'].append(record)
                detail = (format_time(record['median_seconds']) if record['status'] == 'ok'
                          else f"{record['status']}: {record.get('error', '')}")
                print(f"  {case['operation']} bits={case['bits']} exp={case['exponent']}: {detail}",
                      file=sys.stderr, flush=True)
        finally:
            worker.stop(graceful=True)


def print_table(report):
    reports = report['reports']
    width = 22
    print(f"{'operation':<14} {'bits':>12}" + ''.join(f"{r['name']:>{width}}" for r in reports))
    if not reports:
        return
    for i, row in enumerate(reports[0]['rows']):
        operation = row['operation'] + (f" **{row['exponent']}" if row['exponent'] is not None else '')
        text = f"{operation:<14} {row['bits']:>12}"
        for backend in reports:
            if i >= len(backend['rows']):
                cell = 'not run'
            else:
                entry = backend['rows'][i]
                cell = format_time(entry['median_seconds']) if entry['status'] == 'ok' else entry['status']
            text += f'{cell:>{width}}'
        print(text)
    multi = next((r for r in reports if r['name'] == 'tabai-multi'), None)
    single = next((r for r in reports if r['name'] == 'tabai-single'), None)
    if multi is not None:
        print('Distributed NTT calls include warmup; zero means the operation stayed on the primary GPU.')
        for i, row in enumerate(multi['rows']):
            if row['status'] != 'ok':
                continue
            ratio = ''
            if single is not None and i < len(single['rows']) and single['rows'][i]['status'] == 'ok':
                ratio = f", speedup={single['rows'][i]['median_seconds'] / row['median_seconds']:.2f}x"
            print(f"  {row['operation']} bits={row['bits']} exp={row['exponent']}: "
                  f"NTT calls={row['distributed_multiplications']}{ratio}")


def main(argv=None, **defaults):
    args = parse_args(argv, **defaults)
    config = {k: v for k, v in vars(args).items() if k != 'output'}
    report = {'schema_version': 2, 'started_at': datetime.now(timezone.utc).isoformat(),
              'config': config, 'reports': [], 'interrupted': False}
    try:
        run_suite(config, report)
    except KeyboardInterrupt:
        report['interrupted'] = True
        print('Interrupted; worker stopped.', file=sys.stderr)
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'Saved {args.output}', file=sys.stderr)
    print_table(report)
    if report['interrupted']:
        return 130
    return int(any(row['status'] not in ('ok', 'skipped') for r in report['reports'] for row in r['rows']))
