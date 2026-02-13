import pytest
import random
import cupy as cp
from tabai_gpu.core import GPUBigInt
from tabai_gpu.utils import int_to_gpu, gpu_to_int
from tabai_gpu.tabai_int import TabaiInt

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

def test_add_small_numbers(calc):
    for a in range(0,100):
        for b in range(0,100):
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

def test_mul_small_numbers(calc):
    for a in range(0,100):
        for b in range(0,100):
            res = calc.mul(int_to_gpu(a), int_to_gpu(b))
            assert gpu_to_int(res) == a * b

def test_mul_carry_chain(calc):
    a = (1 << 100000) - 1
    b = (1 << 100000) - 1
    res = calc.mul(int_to_gpu(a), int_to_gpu(b))
    assert gpu_to_int(res) == a * b

@pytest.mark.parametrize("a,b", [
    ((1 << 1000) - (1 << 500) + 1, (1 << 999) + (1 << 333) - 1),
    ((1 << 100000) - (1 << 50000) + 1, (1 << 99999) + (1 << 33333) - 1),
], ids=["1000bit", "100000bit"])
def test_large_ops(calc, a, b):
    assert gpu_to_int(calc.add(int_to_gpu(a), int_to_gpu(b))) == a + b
    assert gpu_to_int(calc.sub(int_to_gpu(a), int_to_gpu(b))) == a - b
    assert gpu_to_int(calc.mul(int_to_gpu(a), int_to_gpu(b))) == a * b


class TestNegativeNumbers:
    def test_neg_construction(self):
        assert TabaiInt(-1).to_cpu() == -1
        assert TabaiInt(-100).to_cpu() == -100
        assert TabaiInt(-10**100).to_cpu() == -(10**100)
        assert TabaiInt(0).to_cpu() == 0

    def test_neg_operator(self):
        assert (-TabaiInt(5)).to_cpu() == -5
        assert (-TabaiInt(-5)).to_cpu() == 5
        assert (-TabaiInt(0)).to_cpu() == 0

    def test_abs_operator(self):
        assert abs(TabaiInt(-5)).to_cpu() == 5
        assert abs(TabaiInt(5)).to_cpu() == 5
        assert abs(TabaiInt(0)).to_cpu() == 0

    def test_add_neg_neg(self):
        a, b = -3, -7
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_add_neg_pos(self):
        assert (TabaiInt(-3) + TabaiInt(7)).to_cpu() == 4
        assert (TabaiInt(-7) + TabaiInt(3)).to_cpu() == -4

    def test_add_pos_neg(self):
        assert (TabaiInt(7) + TabaiInt(-3)).to_cpu() == 4
        assert (TabaiInt(3) + TabaiInt(-7)).to_cpu() == -4

    def test_add_cancel_to_zero(self):
        assert (TabaiInt(5) + TabaiInt(-5)).to_cpu() == 0
        assert (TabaiInt(-5) + TabaiInt(5)).to_cpu() == 0

    def test_sub_neg(self):
        assert (TabaiInt(-3) - TabaiInt(7)).to_cpu() == -10
        assert (TabaiInt(3) - TabaiInt(-7)).to_cpu() == 10
        assert (TabaiInt(-3) - TabaiInt(-7)).to_cpu() == 4

    def test_mul_neg(self):
        assert (TabaiInt(-3) * TabaiInt(7)).to_cpu() == -21
        assert (TabaiInt(3) * TabaiInt(-7)).to_cpu() == -21
        assert (TabaiInt(-3) * TabaiInt(-7)).to_cpu() == 21
        assert (TabaiInt(-3) * TabaiInt(0)).to_cpu() == 0
        assert (TabaiInt(0) * TabaiInt(-7)).to_cpu() == 0

    def test_add_neg_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b

    def test_sub_neg_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b

    def test_mul_neg_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_neg_large(self):
        a = -(10**100)
        b = 20**100
        assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b
        assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b
        assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b

    def test_floordiv_neg(self):
        assert (TabaiInt(-7) // TabaiInt(2)).to_cpu() == -7 // 2
        assert (TabaiInt(7) // TabaiInt(-2)).to_cpu() == 7 // -2
        assert (TabaiInt(-7) // TabaiInt(-2)).to_cpu() == -7 // -2
        assert (TabaiInt(-6) // TabaiInt(2)).to_cpu() == -6 // 2

    def test_mod_neg(self):
        assert (TabaiInt(-7) % TabaiInt(2)).to_cpu() == -7 % 2
        assert (TabaiInt(7) % TabaiInt(-2)).to_cpu() == 7 % -2
        assert (TabaiInt(-7) % TabaiInt(-2)).to_cpu() == -7 % -2
        assert (TabaiInt(-6) % TabaiInt(2)).to_cpu() == -6 % 2

    def test_divmod_neg(self):
        for a in [-7, 7, -6, 6, -100, 100]:
            for b in [-3, 3, -2, 2, -1, 1]:
                q, r = divmod(TabaiInt(a), TabaiInt(b))
                eq, er = divmod(a, b)
                assert q.to_cpu() == eq
                assert r.to_cpu() == er

    def test_repr_neg(self):
        assert repr(TabaiInt(-42)) == "TabaiInt(-42)"
        assert repr(TabaiInt(42)) == "TabaiInt(42)"
        assert repr(TabaiInt(0)) == "TabaiInt(0)"


class TestComparison:
    def test_eq(self):
        assert TabaiInt(5) == TabaiInt(5)
        assert TabaiInt(-5) == TabaiInt(-5)
        assert TabaiInt(0) == TabaiInt(0)
        assert not (TabaiInt(5) == TabaiInt(-5))
        assert not (TabaiInt(5) == TabaiInt(3))

    def test_ne(self):
        assert TabaiInt(5) != TabaiInt(3)
        assert TabaiInt(5) != TabaiInt(-5)
        assert not (TabaiInt(5) != TabaiInt(5))

    def test_lt(self):
        assert TabaiInt(-5) < TabaiInt(3)
        assert TabaiInt(-5) < TabaiInt(-3)
        assert TabaiInt(3) < TabaiInt(5)
        assert not (TabaiInt(5) < TabaiInt(3))
        assert not (TabaiInt(5) < TabaiInt(5))

    def test_le(self):
        assert TabaiInt(-5) <= TabaiInt(3)
        assert TabaiInt(5) <= TabaiInt(5)
        assert not (TabaiInt(5) <= TabaiInt(3))

    def test_gt(self):
        assert TabaiInt(5) > TabaiInt(3)
        assert TabaiInt(-3) > TabaiInt(-5)
        assert TabaiInt(3) > TabaiInt(-5)
        assert not (TabaiInt(3) > TabaiInt(5))
        assert not (TabaiInt(5) > TabaiInt(5))

    def test_ge(self):
        assert TabaiInt(5) >= TabaiInt(3)
        assert TabaiInt(5) >= TabaiInt(5)
        assert not (TabaiInt(3) >= TabaiInt(5))

    def test_compare_zero(self):
        assert TabaiInt(0) == TabaiInt(0)
        assert TabaiInt(0) >= TabaiInt(0)
        assert TabaiInt(0) <= TabaiInt(0)
        assert TabaiInt(1) > TabaiInt(0)
        assert TabaiInt(-1) < TabaiInt(0)
        assert TabaiInt(0) > TabaiInt(-1)
        assert TabaiInt(0) < TabaiInt(1)

    def test_compare_large(self):
        a = 10**100
        b = 20**100
        assert TabaiInt(a) < TabaiInt(b)
        assert TabaiInt(-b) < TabaiInt(-a)
        assert TabaiInt(-a) > TabaiInt(-b)
        assert TabaiInt(-b) < TabaiInt(a)

    def test_compare_small_exhaustive(self):
        for a in range(-50, 51):
            for b in range(-50, 51):
                ta, tb = TabaiInt(a), TabaiInt(b)
                assert (ta == tb) == (a == b)
                assert (ta != tb) == (a != b)
                assert (ta < tb) == (a < b)
                assert (ta <= tb) == (a <= b)
                assert (ta > tb) == (a > b)
                assert (ta >= tb) == (a >= b)

    def test_compare_random_large(self):
        random.seed(42)
        for _ in range(50):
            a = random.randint(-(1 << 1000), 1 << 1000)
            b = random.randint(-(1 << 1000), 1 << 1000)
            ta, tb = TabaiInt(a), TabaiInt(b)
            assert (ta == tb) == (a == b)
            assert (ta < tb) == (a < b)
            assert (ta > tb) == (a > b)
