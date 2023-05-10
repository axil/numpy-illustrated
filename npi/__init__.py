import zipfile
from numpy.lib.recfunctions import (
    unstructured_to_structured as u2s,
    structured_to_unstructured as s2u,
)
from itertools import permutations, product

import numpy as np
from numpy.compat import asbytes, asstr, asunicode, os_fspath, os_PathLike, pickle

try:
    from ndfind import find, first_above, first_nonzero
    # print('using ndfind (cython)')
except ImportError:
    from .pyfind import find, first_above, first_nonzero
    # print('using pyfind (python)')

__all__ = (
    "argmin",
    "argmax",
    "nanargmin",
    "nanargmax",
    "T_",
    "sort",
    "savez",
    "savez_compressed",
    "irange",
    "concat",
    "find",
    "first_above",
    "first_nonzero", 
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
    >>> argmin([[4,8,5], [9,3,1]])
    (1, 2)
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


def nanargmin(a):
    """
    Returns the index of the minimum value.
    The result is scalar in 1D case and tuple of indices in 2D and above.
    If the maximum is encountered several times, returns the first match
    in the C order (irrespectively of the order of the array itself).
    E.g.:
    >>> nanargmin([4,3,nan])
    1
    >>> nanargmin([[4,8,5], [9,3,1]])
    (1, 2)
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim > 1:
        return np.unravel_index(np.nanargmin(a), a.shape)
    else:
        return np.nanargmin(a)


def nanargmax(a):
    """
    Returns the index of the maximum value.
    The result is scalar in 1D case and tuple of indices in 2D and above.
    If the maximum is encountered several times, returns the first match
    in the C order (irrespectively of the order of the array itself).
    E.g.:
    >>> nanargmax([nan,5,3])
    1
    >>> nanargmax([[4,3,5], [5,nan,3]])
    (0, 2)
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim > 1:
        return np.unravel_index(np.nanargmax(a), a.shape)
    else:
        return np.nanargmax(a)


def T_(x):
    """
    Returns a view of the array with axes transposed:
      - transposes a matrix just like the original T;
      - transposes 1D array to a 2D column-vector and vica versa;
      - transposes (a less commonly used) 2D row-vector to a 2D column-vector;
      - for 3D arrays and above swaps the last two dimensions.
    E.g.:
    >>> T_(np.array([[1, 2], [3, 4]]))
    array([[1, 3],
           [2, 4]])
    >>> T_(np.array([1, 2, 3]))
    array([[1],
           [2],
           [3]])
    >>> T_(np.array([[1],
                     [2],
                     [3]])
    array([1, 2, 3])
    >>> T_(np.array([[1, 2, 3]]))
    array([[1],
           [2],
           [3]])
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
    """
    Effectively makes npz files appendable.
    A context manager for saving a series of measurements step-by-step.
    If an exception is encountered, gracefully closes the npz file.
    For example the following code
    >>> a1 = [1]; a2 = [2]; a3 = [3]
    >>> with savez('a.npz', compress=True) as fout:
    >>>     fout.write(a1=a1)
    >>>     fout.write(a2=a2)
    >>>     1/0
    >>>     fout.write(a3=a3)
    saves the arrays a1 and a2 to 'a.npz' and closes the file gracefully.
    """

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


class savez_compressed(savez):
    """
    Same context manager as savez with compression enabled by default.
    """
    def __init__(self, file, compress=True, allow_pickle=True, pickle_kwargs=None):
        super().__init__(file, compress, allow_pickle, pickle_kwargs)


def sort(a, by=None, axis=0, ascending=True):
    """
    Rearranges the rows so that the result is sorted by the specified columns
    An extension of `sort` that allows:
      - sorting by column(s)
      - ascending and descending

    If by is a list [c1, c2, ..., cn], sorts by the column c1, resolving the ties using
    the column c2, and so on until cn (just like in pandas). Unlike pandas, the columns
    not present in the `by` argument are used for resolving the remaining ties in the
    left to right order.

    `by=None` is the same as by=[0, 1, 2, ..., a.shape[-1]]

    `ascending` can be either be a scalar or a list.

    For example:
    >>>  sort([[1, 2, 3],
               [3, 1, 5],
               [1, 0, 6]])
    array([[1, 0, 6],
           [1, 2, 3],
           [3, 1, 5]])
    """
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
        a[..., to_negate] *= -1

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
            u[..., to_negate] *= -1
    return u


def irange(start, stop, step=1, dtype=None, tol=1e-6):
    """
    Returns an evenly spaced array from start to stop inclusively.
    If the range `stop-start` is not evenly divisible by step (=if the calculated number 
    of steps is further from the nearest integer than `tol`), raises a ValueError 
    exception.
    """
    if all(isinstance(arg, int) for arg in (start, stop, step)):
        if step > 0:
            return np.arange(start, stop + 1, step, dtype=dtype)
        else:
            return np.arange(start, stop - 1, step, dtype=dtype)
    n = (stop - start) / step
    if abs(round(n) - n) > 1e-6:
        raise ValueError("(stop-start) must be divisible by step")
    return np.linspace(start, stop, round(n) + 1, dtype=dtype)


concat = np.concatenate

