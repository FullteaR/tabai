import pytest


def pytest_addoption(parser):
    parser.addoption("--require-multi-gpu", action="store_true",
                     help="Fail immediately unless at least two CUDA GPUs are visible")


def pytest_sessionstart(session):
    if session.config.getoption("--require-multi-gpu"):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 2:
            raise pytest.UsageError("--require-multi-gpu needs at least two physical CUDA GPUs")
