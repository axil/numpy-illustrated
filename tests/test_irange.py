import pytest
import numpy as np
from math import isclose, pi

from npi import irange


def test1():
    assert np.array_equal(irange(1, 3), [1, 2, 3])
    assert np.array_equal(irange(3, 1, -1), [3, 2, 1])
    assert np.array_equal(irange(0, 1, 0.1), np.arange(0, 1.01, 0.1))
    assert np.allclose(irange(1, 0, -0.1), np.arange(1, -0.01, -0.1))


def test2():
    assert np.issubdtype(irange(1, 3).dtype, np.integer)
    assert np.issubdtype(irange(1.0, 3.0).dtype, np.floating)
    assert np.issubdtype(irange(1, 3, 0.1).dtype, np.floating)
    assert np.issubdtype(irange(1, 3, dtype=np.uint8).dtype, np.uint8)
    assert np.issubdtype(irange(1, 3, dtype=np.float16).dtype, np.float16)


def test3():
    with pytest.raises(ValueError):
        irange(0, 1, 0.3)
    assert np.allclose(irange(0, 1, 0.3, raises=False), [0, 0.3, 0.6, 0.9])


def test4():
    a = irange(-pi, pi, pi / 100)
    assert len(a) == 201
    assert isclose(a[0], -pi)
    assert isclose(a[-1], pi)
    assert isclose(a[1] - a[0], pi / 100)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
