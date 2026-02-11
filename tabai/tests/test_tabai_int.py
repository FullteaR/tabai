import pytest
from tabai_gpu import TabaiInt


def test_add_1_1():
    assert (TabaiInt(1) + TabaiInt(1)).to_cpu() == 2


def test_add_basic():
    a, b = 10**100, 20**100
    assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b


def test_add_small_numbers():
    for a in range(0, 100):
        for b in range(0, 100):
            assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b


def test_add_carry_chain():
    a = (1 << 100000) - 1
    assert (TabaiInt(a) + TabaiInt(1)).to_cpu() == 1 << 100000


def test_sub_basic():
    a, b = 20**100, 10**100
    assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b


def test_mul_1_1():
    assert (TabaiInt(1) * TabaiInt(1)).to_cpu() == 1


def test_mul_basic():
    a, b = 10**100, 20**100
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b


def test_mul_by_zero():
    assert (TabaiInt(10**100) * TabaiInt(0)).to_cpu() == 0


def test_mul_by_one():
    a = 10**100
    assert (TabaiInt(a) * TabaiInt(1)).to_cpu() == a


def test_mul_small_numbers():
    for a in range(0, 100):
        for b in range(0, 100):
            assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b


def test_mul_carry_chain():
    a = (1 << 100000) - 1
    b = (1 << 100000) - 1
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b


def test_repr():
    assert repr(TabaiInt(42)) == "TabaiInt(42)"


def test_from_int_and_to_cpu_roundtrip():
    values = [0, 1, 2**32 - 1, 2**32, 2**64, 10**100]
    for v in values:
        assert TabaiInt(v).to_cpu() == v


@pytest.mark.parametrize("a,b", [
    ((1 << 1000) - (1 << 500) + 1, (1 << 999) + (1 << 333) - 1),
    ((1 << 100000) - (1 << 50000) + 1, (1 << 99999) + (1 << 33333) - 1),
], ids=["1000bit", "100000bit"])
def test_large_ops(a, b):
    assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b
    assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b
