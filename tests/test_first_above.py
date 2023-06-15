import pytest
import numpy as np
from itertools import permutations, product
from decimal import Decimal as D

from npi import first_above


def test0():
    assert first_above([3, 1, 4, 1, 5], 4) == 4
    assert first_above([1, 2, 3], 7) == -1
    assert first_above([1.1, 1.2, 1.3], 1.2) == 2


def test1():
    assert first_above([1, 3, 4, 7, 9], 1) == 1
    assert first_above([1, 3, 4, 7, 9], 4) == 3
    assert first_above([1, 3, 4, 7, 9], 9) == -1
    assert first_above([1, 3, 4, 7, 9], 5) == 3
    assert first_above([1, 3, 4, 7, 9], 10) == -1
    assert first_above([1, 3, 4, 7, 9], 0) == 0


def test2():
    a = [1, 3, 4, 7, 9]
    assert first_above(a, 1.0) == 1
    assert first_above(a, 4.0) == 3
    assert first_above(a, 9.0) == -1
    assert first_above(a, 5.0) == 3
    assert first_above(a, 10.0) == -1
    assert first_above(a, 0.0) == 0


def test3():
    a = [1.0, 3.0, 4.0, 7.0, 9.0]
    assert first_above(a, 1.0) == 1
    assert first_above(a, 4.0) == 3
    assert first_above(a, 9.0) == -1
    assert first_above(a, 5.0) == 3
    assert first_above(a, 10.0) == -1
    assert first_above(a, 0.0) == 0


def test4():
    a = [1, 3, 4, 7, 9]
    assert first_above(a, 1.0, sorted=True) == 1
    assert first_above(a, 4.0, sorted=True) == 3
    assert first_above(a, 9.0, sorted=True) == -1
    assert first_above(a, 5.0, sorted=True) == 3
    assert first_above(a, 10.0, sorted=True) == -1
    assert first_above(a, 0.0, sorted=True) == 0


def test_complex():
    with pytest.raises(ValueError):
        first_above([1j], 2)
    with pytest.raises(ValueError):
        first_above([1], 2j)


def test_bool():
    with pytest.raises(ValueError):
        first_above([True], 2)
    with pytest.raises(ValueError):
        first_above([1], True)


INTS_AND_FLOATS = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
]


def test_int_float():
    for a_type in INTS_AND_FLOATS:
        for v_type in INTS_AND_FLOATS + [int, float]:
            a = np.array([0, 1, 2, 3], dtype=a_type)
            v = v_type(1)
            assert first_above(a, v, sorted=True) == 2, (a_type, v_type)


def test_other_datatypes():
    assert first_above(np.array(["a", "bb", "ccc"]), "bb") == 2

    a = np.arange(np.datetime64("2023-01-20"), np.datetime64("2023-01-23"))
    assert first_above(a, np.datetime64("2023-01-21")) == 2

    assert first_above(np.array([D(1), D(2), D(3)]), D(2)) == 2
    assert first_above(np.array([D(1), D(2), D(3)]), D(2), sorted=True) == 2


def test_float16():
    a = np.arange(0.0, 10.0, dtype=np.float16)
    v = np.float16(3.0)
    assert first_above(a, v) == 4
    assert first_above(a, v, sorted=True) == 4


def test_first_above():
    a = [1, 2, 3, 3, 3, 4]
    assert first_above(a, 0) == 0
    assert first_above(a, 1) == 1
    assert first_above(a, 2) == 2
    assert first_above(a, 3) == 5
    assert first_above(a, 4) == -1
    assert first_above(a, 5) == -1


def test_first_above_sorted():
    a = [1, 2, 3, 3, 3, 4]
    assert first_above(a, 0, sorted=True) == 0
    assert first_above(a, 1, sorted=True) == 1
    assert first_above(a, 2, sorted=True) == 2
    assert first_above(a, 3, sorted=True) == 5
    assert first_above(a, 4, sorted=True) == -1
    assert first_above(a, 5, sorted=True) == -1


def test_first_above1():
    a = [1.0, 2.0, 3.0, 3.0, 3.0, 4.0]
    assert first_above(a, 0) == 0
    assert first_above(a, 1) == 1
    assert first_above(a, 2) == 2
    assert first_above(a, 3) == 5
    assert first_above(a, 4) == -1
    assert first_above(a, 5) == -1


def test_raises():
    with pytest.raises(ValueError):
        first_above([1, 2, 3], 7, raises=True)


SIGNED = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED = [np.uint8, np.uint16, np.uint32, np.uint64]


def test_mixed():
    for t1 in SIGNED:
        for t2 in UNSIGNED:
            assert first_above(np.array([-1, 1, 2], dtype=t1), t2(1)) == 2

    for t1 in UNSIGNED:
        for t2 in SIGNED:
            assert first_above(np.array([2, 3, 5], dtype=t1), t2(-1)) == 0


def test_ndarray():
    with pytest.raises(ValueError):
        first_above([[1, 2], [3, 4]], 5)


if __name__ == "__main__":
    pytest.main(["-s", "-x", __file__])  # + '::test7'])
    # pytest.main(["-s", __file__])  # + '::test7'])
