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


def test_floordiv_basic():
    a, b = 7, 3
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == 2


def test_mod_basic():
    a, b = 7, 3
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == 1


def test_divmod_basic():
    a, b = 7, 3
    q, r = divmod(TabaiInt(a), TabaiInt(b))
    assert q.to_cpu() == 2
    assert r.to_cpu() == 1


def test_div_exact():
    a, b = 6, 3
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == 2
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == 0


def test_div_small_numbers():
    for a in range(0, 100):
        for b in range(1, 100):
            assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == a // b
            assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == a % b


def test_div_zero_dividend():
    assert (TabaiInt(0) // TabaiInt(5)).to_cpu() == 0
    assert (TabaiInt(0) % TabaiInt(5)).to_cpu() == 0


def test_div_by_one():
    a = 10**100
    assert (TabaiInt(a) // TabaiInt(1)).to_cpu() == a
    assert (TabaiInt(a) % TabaiInt(1)).to_cpu() == 0


def test_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        TabaiInt(7) // TabaiInt(0)


def test_div_large():
    a, b = 20**100, 10**100
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == a // b
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == a % b


@pytest.mark.parametrize("a,b", [
    ((1 << 1000) - (1 << 500) + 1, (1 << 999) + (1 << 333) - 1),
    ((1 << 100000) - (1 << 50000) + 1, (1 << 99999) + (1 << 33333) - 1),
], ids=["1000bit", "100000bit"])
def test_large_ops(a, b):
    assert (TabaiInt(a) + TabaiInt(b)).to_cpu() == a + b
    assert (TabaiInt(a) - TabaiInt(b)).to_cpu() == a - b
    assert (TabaiInt(a) * TabaiInt(b)).to_cpu() == a * b
    assert (TabaiInt(a) // TabaiInt(b)).to_cpu() == a // b
    assert (TabaiInt(a) % TabaiInt(b)).to_cpu() == a % b


def test_pow_basic():
    assert (TabaiInt(2) ** TabaiInt(5)).to_cpu() == 32
    assert (TabaiInt(2) ** 4).to_cpu() == 16
    assert (3 ** TabaiInt(5)).to_cpu() == 243


def test_pow_zero_exponent():
    assert (TabaiInt(5) ** 0).to_cpu() == 1
    assert (TabaiInt(0) ** 0).to_cpu() == 1


def test_pow_one_exponent():
    assert (TabaiInt(42) ** 1).to_cpu() == 42


def test_pow_zero_base():
    assert (TabaiInt(0) ** 5).to_cpu() == 0


def test_pow_one_base():
    assert (TabaiInt(1) ** 100).to_cpu() == 1


def test_pow_negative_exponent():
    with pytest.raises(ValueError):
        TabaiInt(2) ** TabaiInt(-3)


def test_pow_small_numbers():
    for a in range(0, 20):
        for b in range(0, 10):
            assert (TabaiInt(a) ** TabaiInt(b)).to_cpu() == a ** b


def test_pow_large():
    a, b = 10**100, 3
    assert (TabaiInt(a) ** TabaiInt(b)).to_cpu() == a ** b
