import numpy as np

from npi import sort

def test1():
    a = np.array([[2, 4, 2, 1, 1],
       [1, 4, 1, 3, 1],
       [2, 1, 2, 3, 1],
       [4, 1, 1, 3, 1],
       [1, 0, 2, 1, 3]])

    assert sort(a) == 