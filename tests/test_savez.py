import pytest

from npi import savez
import tempfile
from pathlib import Path

import numpy as np

a = np.eye(3)
b = np.eye(5)
c = np.eye(7)


def test1():
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname) / 'a.npz'
        with savez(filename, compress=True) as fout:
            fout.write(a, b)
            fout.write(c)

        z = np.load(filename)
        assert list(z.keys()) == ['arr_0', 'arr_1', 'arr_2']

        assert np.array_equal(z['arr_0'], a)
        assert np.array_equal(z['arr_1'], b)
        assert np.array_equal(z['arr_2'], c)
        z.close()

def test2():
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname) / 'a.npz'
        with savez(filename, compress=True) as fout:
            fout.write(a, b=b)
            fout.write(c)

        z = np.load(filename)
        assert set(list(z.keys())) == set(['arr_0', 'b', 'arr_1'])

        assert np.array_equal(z['arr_0'], a)
        assert np.array_equal(z['b'], b)
        assert np.array_equal(z['arr_1'], c)
        z.close()


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
