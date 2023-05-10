import pytest
import numpy as np
from itertools import permutations, product
from decimal import Decimal as D

from npi import first_nonzero

NUMERICS = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


def test_int_float():
    for a_type in NUMERICS:
        a = np.array([0, 0, 1, 0, 3], dtype=a_type)
        assert first_nonzero(a) == 2, a_type


def test_other_datatypes():
    assert first_nonzero(np.array(["", "", "a", "", "b"])) == 2
    assert first_nonzero(np.array([D(0), D(0), D(1), D(0), D(3)])) == 2


def test_missing():
    assert first_nonzero([0, 0, 0]) == -1
    assert first_nonzero([0, 0, 0], missing=None) is None
    with pytest.raises(ValueError):
        first_nonzero([0, 0, 0], raises=True)


def test_raises():
    with pytest.raises(ValueError):
        first_nonzero([[1, 2], [3, 4]])


if __name__ == "__main__":
    pytest.main(["-s", "-x", __file__])  # + '::test7'])
    # pytest.main(["-s", __file__])  # + '::test7'])
