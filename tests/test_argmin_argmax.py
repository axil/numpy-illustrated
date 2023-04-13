import pytest
import numpy as np
from npi import argmin, argmax

def test1():
    assert argmin([4,3,5]) == 1
    assert argmax([4,5,3]) == 1

    assert argmin([[4,5,3], [3,4,5]]) == (0, 2)
    assert argmax([[4,3,5], [5,4,3]]) == (0, 2)
    
def test2():
    assert argmin([4,2,10,3]) == 1
    assert argmin(np.array([4,2,10,3])) == 1
    
    assert argmax([4,2,10,3]) == 2
    assert argmax(np.array([4,2,10,3])) == 2


def test3():
    a = np.array([[3, 1, 3, 3],
       [4, 2, 0, 3],
       [0, 2, 4, 1]])
    assert argmin(a) == (1, 2)
    assert argmax(a) == (1, 0) 


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])

