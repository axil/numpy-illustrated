import pytest

import numpy as np

from npi import T

@pytest.mark.parametrize('x, y', [
    ([1], [[1]]),
    ([1,2,3], [[1],[2],[3]]),
    ([[1],[2],[3]], [1,2,3]),
    ([[1,2],[3,4]], [[1,3],[2,4]]),
    (np.zeros((2,3,4)), np.zeros((2,4,3))),
])
def test1(x, y):
    assert np.array_equal(T(x), np.array(y))
    assert np.array_equal(T(tuple(x)), np.array(y))
    assert np.array_equal(T(np.array(x)), np.array(y))

def test2():
    with pytest.raises(ValueError):
        assert T('abc')


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
