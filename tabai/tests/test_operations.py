import pytest
import cupy as cp
import random
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

@pytest.mark.parametrize("bits", [1000, 100000])
def test_random_ops(calc, bits):
    a = random.getrandbits(bits)
    b = random.getrandbits(bits)
    # 加算
    assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b
    # 減算 (a > b に調整)
    if a < b: a, b = b, a
    assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b
