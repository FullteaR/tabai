import pytest
import cupy as cp
from tabai_gpu.core import GPUBigInt
from tabai_gpu.utils import int_to_gpu, gpu_to_int

@pytest.fixture
def calc():
    return GPUBigInt()

def test_add_1_1(calc):
    a, b = 1, 1
    res = calc.add(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == 2

def test_add_basic(calc):
    a, b = 10**100, 20**100
    res = calc.add(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a + b

def test_add_carry_chain(calc):
    a = (1 << 100000) - 1
    b = 1
    res = calc.add(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == (1 << 100000)

def test_sub_basic(calc):
    a, b = 20**100, 10**100
    res = calc.sub(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a - b

def test_mul_1_1(calc):
    res = calc.mul(int_to_gpu(1), int_to_gpu(1))
    assert gpu_to_int(res) == 1

def test_mul_basic(calc):
    a, b = 10**100, 20**100
    res = calc.mul(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a * b

def test_mul_by_zero(calc):
    a = 10**100
    res = calc.mul(int_to_gpu(a), int_to_gpu(0))
    assert gpu_to_int(res) == 0

def test_mul_by_one(calc):
    a = 10**100
    res = calc.mul(int_to_gpu(a), int_to_gpu(1))
    assert gpu_to_int(res) == a

def test_mul_carry_chain(calc):
    a = (1 << 100000) - 1
    b = (1 << 100000) - 1
    res = calc.mul(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a * b

@pytest.mark.parametrize("a,b", [
    ((1 << 1000) - (1 << 500) + 1, (1 << 999) + (1 << 333) - 1),
    ((1 << 100000) - (1 << 50000) + 1, (1 << 99999) + (1 << 33333) - 1),
])
def test_large_ops(calc, a, b):
    assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b
    assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b
    assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b
