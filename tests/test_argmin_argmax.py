import pytest
import numpy as np
from npi import argmin, argmax

def test1():
    assert argmin([4,2,10,3]) == 1
    assert argmin(np.array([4,2,10,3])) == 1
    assert argmax([4,2,10,3]) == 2
    assert argmax(np.array([4,2,10,3])) == 2


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])

