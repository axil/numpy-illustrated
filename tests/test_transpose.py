import pytest

import numpy as np

from npi import T_

@pytest.mark.parametrize('x, y', [
    ([1], [[1]]),
    ([1,2,3], [[1],[2],[3]]),
    ([[1],[2],[3]], [1,2,3]),
    ([[1,2],[3,4]], [[1,3],[2,4]]),
    (np.zeros((2,3,4)), np.zeros((2,4,3))),
])
def test1(x, y):
    assert np.array_equal(T_(x), np.array(y))
    assert np.array_equal(T_(tuple(x)), np.array(y))
    assert np.array_equal(T_(np.array(x)), np.array(y))

def test2():
    assert T_('abc') == np.array('abc')


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
