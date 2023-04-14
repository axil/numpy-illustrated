import zipfile
from numpy.lib.recfunctions import (
    unstructured_to_structured as u2s,
    structured_to_unstructured as s2u,
)
from itertools import permutations, product

import numpy as np
from numpy.compat import asbytes, asstr, asunicode, os_fspath, os_PathLike, pickle

__all__ = (
    "argmin",
    "argmax",
    "T_",
    "sort",
    "savez",
    "irange",
)


def argmin(a):
    """
    Returns the index of the minimum value.
    The result is scalar in 1D case and tuple of indices in 2D and above.
    If the maximum is encountered several times, returns the first match
    in the C order (irrespectively of the order of the array itself).
    E.g.:
    >>> argmin([4,3,5])
    1
    >>> argmin([[4,5,3], [3,4,5]])
    (0, 2)
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim > 1:
        return np.unravel_index(np.argmin(a), a.shape)
    else:
        return np.argmin(a)


def argmax(a):
    """
    Returns the index of the maximum value.
    The result is scalar in 1D case and tuple of indices in 2D and above.
    If the maximum is encountered several times, returns the first match
    in the C order (irrespectively of the order of the array itself).
    E.g.:
    >>> argmax([4,5,3])
    1
    >>> argmax([[4,3,5], [5,4,3]])
    (0, 2)
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim > 1:
        return np.unravel_index(np.argmax(a), a.shape)
    else:
        return np.argmax(a)


def T_(x):
    """
    Returns a view of the array with axes transposed.
    Behaves just like original T except it transposes 1D array to
    a column-vector and vica versa. E.g.:
    >>> T_(np.array([1,2,3]))
    array([[1],
           [2],
           [3]])
    >>> T_(np.array([[1],
                     [2],
                     [3]])
    array([1, 2, 3])
    """
    x = np.array(x)
    if x.ndim == 0:
        return x
    elif x.ndim == 1:
        return x[:, None]
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x[:, 0]
        else:
            return x.T
    else:
        return np.swapaxes(x, x.ndim - 2, x.ndim - 1)


def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.
    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    if not hasattr(file, "read"):
        file = os_fspath(file)
    import zipfile

    kwargs["allowZip64"] = True
    return zipfile.ZipFile(file, *args, **kwargs)


class savez:
    def __init__(self, file, compress=False, allow_pickle=True, pickle_kwargs=None):
        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        self.file = file
        self.compress = compress
        self.allow_pickle = allow_pickle
        self.pickle_kwargs = pickle_kwargs
        self.idx = 0

    def __enter__(self):
        if not hasattr(self.file, "write"):
            self.file = os_fspath(self.file)
            if not self.file.endswith(".npz"):
                self.file = self.file + ".npz"

        if self.compress:
            compression = zipfile.ZIP_DEFLATED
        else:
            compression = zipfile.ZIP_STORED

        self.zipf = zipfile_factory(self.file, mode="w", compression=compression)
        return self

    def write(self, *args, **kwargs):
        namedict = kwargs
        for i, val in enumerate(args, start=self.idx):
            key = "arr_%d" % i
            if key in namedict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
        self.idx += i + 1

        for key, val in namedict.items():
            fname = key + ".npy"
            val = np.asanyarray(val)
            # always force zip64, gh-10776
            with self.zipf.open(fname, "w", force_zip64=True) as fid:
                np.lib.format.write_array(
                    fid,
                    val,
                    allow_pickle=self.allow_pickle,
                    pickle_kwargs=self.pickle_kwargs,
                )

    def __exit__(self, *args):
        self.zipf.close()


def sort(a, by=None, axis=0, ascending=True):
    if isinstance(by, (list, tuple)):
        order = [f"f{field}" for field in by]
    elif isinstance(by, int):
        order = f"f{by}"
    elif by is None:
        order = None
    else:
        raise TypeError(f"Unsupported `by` type: {type(by)}")

    if isinstance(ascending, (list, tuple, np.ndarray)):
        if len(ascending) == 1:
            asc = bool(ascending[0])
        elif len(ascending) == len(by):
            asc = list(ascending)
        else:
            raise ValueError(
                f"Length of `ascending`({len(ascending)}) != length of `by`({len(by)})."
            )
    elif isinstance(ascending, (bool, int)):
        asc = bool(ascending)

    # invert columns
    a = np.array(a)
    if asc is False:
        a *= -1
    elif asc is True:
        pass
    else:
        to_negate = []
        for field, asc1 in zip(by, asc):
            if asc1 is False:
                to_negate.append(field)
        a[:, to_negate] *= -1

    # sort
    if a.ndim > 1:
        s = u2s(a)
        s.sort(order=order)
        u = s2u(s)
    elif a.ndim == 1:
        a.sort()
        u = a
    else:
        pass

    # invert columns back
    if asc is False:
        u *= -1
    elif asc is True:
        pass
    else:
        if to_negate:
            u[:, to_negate] *= -1
    return u


def irange(start, stop, step=1):
    if all(isinstance(arg, int) for arg in (start, stop, step)):
        if step > 0:
            return np.arange(start, stop + 1, step)
        else:
            return np.arange(start, stop - 1, step)
    n = (stop - start) / step
    if abs(int(n) - n) > 1e-6:
        raise ValueError("(stop-start) must be divisible by step")
    return np.linspace(start, stop, int(n) + 1)
