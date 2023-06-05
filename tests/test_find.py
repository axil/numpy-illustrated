import pytest
import numpy as np
from itertools import permutations, product
from decimal import Decimal as D

from npi import find


def test0():
    assert find([3, 1, 4, 1, 5], 4) == 2
    assert find([1, 2, 3], 7) == -1
    assert find([1.1, 1.2, 1.3], 1.2) == 1
    assert find(np.arange(0, 1, 0.1), 0.3) == 3
    assert find([[3, 8, 4], [5, 2, 7]], 7) == (1, 2)
    assert find([[3, 8, 4], [5, 2, 7]], 9) == -1


def test1():
    assert find([1, 3, 4, 7, 9], 1) == 0
    assert find([1, 3, 4, 7, 9], 4) == 2
    assert find([1, 3, 4, 7, 9], 9) == 4
    assert find([1, 3, 4, 7, 9], 5) == -1
    assert find([1, 3, 4, 7, 9], 10) == -1
    assert find([1, 3, 4, 7, 9], 0) == -1


def test2():
    a = [1, 3, 4, 7, 9]
    assert find(a, 1.0) == 0
    assert find(a, 4.0) == 2
    assert find(a, 9.0) == 4
    assert find(a, 5.0) == -1
    assert find(a, 10.0) == -1
    assert find(a, 0.0) == -1


def test3():
    a = [1.0, 3.0, 4.0, 7.0, 9.0]
    assert find(a, 1.0) == 0
    assert find(a, 4.0) == 2
    assert find(a, 9.0) == 4
    assert find(a, 5.0) == -1
    assert find(a, 10.0) == -1
    assert find(a, 0.0) == -1


def test4():
    a = [1, 3, 4, 7, 9]
    assert find(a, 1.0, sorted=True) == 0
    assert find(a, 4.0, sorted=True) == 2
    assert find(a, 9.0, sorted=True) == 4
    assert find(a, 5.0, sorted=True) == -1
    assert find(a, 10.0, sorted=True) == -1
    assert find(a, 0.0, sorted=True) == -1


def test5():
    a1 = np.arange(0, 0.5, 0.1)
    assert find(a1, 0.3) == 3
    a2 = np.arange(0.5, 1, 0.1)
    assert find(a2, 0.8) == 3


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
    np.longdouble,
]


def test_int_float():
    for a_type in INTS_AND_FLOATS:
        for v_type in INTS_AND_FLOATS + [int, float]:
            a = np.array([0, 1, 2, 3], dtype=a_type)
            v = v_type(2)
            assert find(a, v) == 2, (a_type, v_type)
            assert find(a, v, sorted=True) == 2, (a_type, v_type)


def test_datatypes():
    for a_type in (complex, float, int):
        for v_type in (complex, float, int):
            a = np.array([0, 1, 2, 1], dtype=a_type)
            v = v_type(1)
            assert find(a, v) == 1
            v = v_type(3)
            assert find(a, v) == -1

    for a_type in (float, int):
        for v_type in (float, int):
            a = np.array([0, 1, 2, 3], dtype=a_type)
            v = v_type(1)
            assert find(a, v, sorted=True) == 1
            v = v_type(4)
            assert find(a, v, sorted=True) == -1

    assert find([1., 2., 3.], np.inf) == -1
    assert find([1., 2., 3.], np.inf, sorted=True) == -1
    
    assert find([1., 2., 3.], np.nan) == -1
    assert find([1., 2., 3.], np.nan, sorted=True) == -1


    assert find([False, False, True, False, True], True) == 2
    assert find([False, False, True, False, True], False) == 0
    assert find([False, False], True) == -1
    assert find([True, True], False) == -1

    for a_type in (complex, float, int):
        for v_type in (complex, float, int):
            a = np.array([[0, 1, 5], [5, 4, 5]], dtype=a_type)
            v = v_type(5)
            assert find(a, v) == (0, 2)

    for a_type in (complex, float, int, bool):
        for v_type in (complex, float, int, bool):
            a = np.array([[0, 1, 2], [1, 4, 5]], dtype=a_type)
            v = v_type(1)
            assert find(a, v) == (0, 1)

    assert find(np.array([1 + 1j, 2 + 2j, 3 + 3j]), 2 + 2j) == 1

    with pytest.raises(ValueError):
        find(np.array([1 + 1j, 2 + 2j, 3 + 3j]), 2 + 2j, sorted=True)

    assert find(np.array(["a", "bb", "ccc"]), "bb") == 1

    a = np.arange(np.datetime64("2023-01-20"), np.datetime64("2023-01-23"))
    a
    assert find(a, np.datetime64("2023-01-22")) == 2

    assert find(np.array([D(1), D(2), D(3)]), D(2)) == 1
    assert find(np.array([D(1), D(2), D(3)]), D(2), sorted=True) == 1

    assert find(np.array([[1, 2], [3, np.nan], [np.nan, 4]]), np.nan) == (1, 1)


def test_float16():
    a = np.arange(0, 1, 0.1, dtype=np.float16)
    v = np.float16(0.3)
    assert find(a, v) == -1
    assert find(a, v, sorted=True) == -1
    assert find(a, v, rtol=1e-2) == 3
    assert find(a, v, rtol=1e-2, sorted=True) == 3


def test_special_floats():
    a = np.array(
        [0.0, 2.0, np.nan, np.inf, np.inf, np.NINF, np.nan, np.NZERO, np.PZERO]
    )
    a
    assert find(a, np.NZERO) == 0
    assert find(a, np.PZERO) == 0
    assert find(a, 0.0) == 0
    assert find(a, np.nan) == 2
    assert find(a, np.inf) == 3
    assert find(a, np.NINF) == 5
    a.sort()
    assert find(a, np.NINF) == 0
    assert find(a, np.inf) == 5
    assert find(a, np.nan) == 7
    assert find(a, np.NZERO) == 1
    assert find(a, np.PZERO) == 1
    assert find(a, 0.0) == 1
    assert find(a, np.NZERO, sorted=True) == 1
    assert find(a, np.PZERO, sorted=True) == 1
    assert find(a, 0.0, sorted=True) == 1
    assert find(a, np.NINF, sorted=True) == 0
    assert find(a, np.inf, sorted=True) == 5
    assert find(a, np.nan, sorted=True) == 7


def test_special_complex():
    a = np.array([0.0, 2j, np.nan, np.inf, np.inf, np.NINF, np.nan, np.NZERO, np.PZERO])
    assert find(a, np.NZERO) == 0
    assert find(a, np.PZERO) == 0
    assert find(a, 0.0) == 0
    assert find(a, np.nan) == 2
    assert find(a, np.inf) == 3
    assert find(a, np.NINF) == 5
    a.sort()
    assert find(a, np.NINF) == 0
    assert find(a, np.inf) == 5
    assert find(a, np.nan) == 7
    assert find(a, np.NZERO) == 1
    assert find(a, np.PZERO) == 1
    assert find(a, 0.0) == 1


def test_special_datetime():
    a = np.arange(np.datetime64("2023-01-20"), np.datetime64("2023-01-23"))
    a1 = np.hstack([a, np.datetime64("nat"), a, np.datetime64("nat"), a])
    find(a, np.datetime64("nat")) == -1
    find(a1, np.datetime64("nat")) == 3


def test_mixed_types():
    assert find(np.array([1, 2, 3], dtype=np.uint8), np.int32(1000)) == -1
    assert find(np.array([1, 2, 3], dtype=np.uint8), np.int32(1000), sorted=True) == -1
    
    a = np.array([D(1), D(2), D(3)])
    assert find(a, 2.0) == 1
    assert find(a, 2.00000001) == -1
    assert find(a, 2.0, sorted=True) == 1
    assert find(a, 2.00000001, sorted=True) == -1

    a = np.array([1, 2, 3])
    assert find(a, 2.0) == 1
    assert find(a, 2.000000001) == 1
    assert find(a, 2.0, sorted=True) == 1
    assert find(a, 2.000000001, sorted=True) == 1

    a = np.array([D(1), D(2), D(3)])
    assert find(a, np.nan) == -1

    a = np.array([D(1), D(2), D(3), np.nan])
    assert find(a, np.nan) == 3


def test_signed_unsigned():
    assert find(np.array([2**62], np.int64), np.uint64(2**62)) == 0
    assert find(np.array([2**62], np.uint64), np.int64(2**62)) == 0
    assert find(np.array([2**62], np.int64), np.uint64(2**62 + 1)) == -1
    assert find(np.array([2**62], np.uint64), np.int64(2**62 + 1)) == -1
    assert find(np.array([2**64 - 1], np.uint64), np.int64(-1)) == -1


if __name__ == "__main__":
#    test_special_complex()
    pytest.main(["-s", "-x", __file__])  # + '::test7'])
#    pytest.main(["-s", __file__  + '::test_special_complex'])
