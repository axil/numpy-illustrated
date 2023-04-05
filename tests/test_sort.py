import pytest
import numpy as np
from itertools import permutations, product
import pandas as pd

from npi import sort

def test0():
    assert np.array_equal(sort([1,3,5,2]), np.array([1, 2, 3, 5]))
    assert np.array_equal(sort([1,3,5,2], ascending=False), np.array([5, 3, 2, 1]))

def test1():
    a = np.array([[2, 4, 2, 1, 1],
       [1, 4, 1, 3, 1],
       [2, 1, 2, 3, 1],
       [4, 1, 1, 3, 1],
       [1, 0, 2, 1, 3]])

    assert np.array_equal(sort(a), np.array([[1, 0, 2, 1, 3],
       [1, 4, 1, 3, 1],
       [2, 1, 2, 3, 1],
       [2, 4, 2, 1, 1],
       [4, 1, 1, 3, 1]]))

def test2():
    a = np.array(
        [[2, 4, 2, 1, 1],
         [1, 4, 1, 3, 1],
         [2, 1, 2, 3, 1],
         [4, 1, 1, 3, 1],
         [1, 0, 2, 1, 3]])

    for i in range(5):
        for p in permutations(np.arange(0, 5), i):
            x = sort(a, list(p))
            y = pd.DataFrame(x.copy()).sort_values(list(p)).values
            assert np.array_equal(x, y), (x, y)

def test3():
    a = np.array(
        [[2, 4, 2, 1, 1],
         [1, 4, 1, 3, 1],
         [2, 1, 2, 3, 1],
         [4, 1, 1, 3, 1],
         [1, 0, 2, 1, 3]])

    for i in range(5):
        for p in permutations(np.arange(0, 5), i):
            for asc in [True, False] + list(product([True, False], repeat=i)):
                x = sort(a, list(p), ascending=asc)
                y = pd.DataFrame(x.copy()).sort_values(list(p), ascending=asc).values
                assert np.array_equal(x, y), (p, asc, x, y)

def est4():                                                       
    a = np.array([[[3, 4, 2, 4],
        [3, 1, 2, 2],
        [3, 4, 3, 2]],

       [[3, 1, 3, 1],
        [3, 4, 0, 3],
        [1, 4, 3, 0]]])

    assert np.array_equal(sort(a), [[[3, 1, 2, 2],
        [3, 4, 2, 4],
        [3, 4, 3, 2]],

       [[1, 4, 3, 0],
        [3, 1, 3, 1],
        [3, 4, 0, 3]]])

    assert np.array_equal(sort(a, by=[2,1]), [[[3, 1, 2, 2],
        [3, 4, 2, 4],
        [3, 4, 3, 2]],

       [[3, 4, 0, 3],
        [3, 1, 3, 1],
        [1, 4, 3, 0]]])

    assert np.array_equal(sort(a, by=[2,1], ascending=[False, True]), 
       np.array([[[3, 4, 3, 2],
        [3, 1, 2, 2],
        [3, 4, 2, 4]],

       [[3, 1, 3, 1],
        [1, 4, 3, 0],
        [3, 4, 0, 3]]]))

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
