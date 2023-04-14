import pytest
import numpy as np

from npi import irange


def test1():
    assert np.array_equal(irange(1, 3), [1, 2, 3])
    assert np.array_equal(irange(3, 1, -1), [3, 2, 1])
    assert np.array_equal(irange(0, 1, 0.1), np.arange(0, 1.01, 0.1))
    assert np.allclose(irange(1, 0, -0.1), np.arange(1, -0.01, -0.1))


def test2():
    assert np.issubdtype(irange(1, 3).dtype, np.integer)
    assert np.issubdtype(irange(1.0, 3.0).dtype, np.floating)


def test3():
    with pytest.raises(ValueError):
        irange(0, 1, 0.3)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
