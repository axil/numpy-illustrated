import numpy as np


def _generic_find(a, v, sorted=False):
    """
    a ndarray with dtype in (int, bool, string, bytes, datetime64, object)
    v scalar with type in (int, bool, string, bytes, datetime64, object)
    """
    if sorted:
        i = np.searchsorted(a, v)
        if i == a.shape[0] or a[i] != v:
            return -1
        else:
            return i
    else:
        indices = np.where(a == v)
        if len(indices[0]):
            if a.ndim == 1:
                return indices[0][0]
            else:
                return next(zip(*np.where(a == v)))
        else:
            return -1


def _generic_float_find(a, v, sorted=False):
    """
    a ndarray with dtype in (complex, float, int, bool, string, bytes, datetime64, object)
    v is nan, inf or NINF
    """
    if sorted:
        i = np.searchsorted(a, v)
        if i == a.shape[0]:
            return -1
        elif np.isnan(v) or a[i] == v:
            return i
    else:
        if np.isnan(v):
            indices = np.where(np.isnan(a))
        else:
            indices = np.where(a == v)
        if len(indices[0]):
            if a.ndim == 1:
                return indices[0][0]
            else:
                return next(zip(*indices))
        else:
            return -1


def _nan_find(a, sorted=False):
    """
    a ndarray with dtype == object
    v is nan
    """
    if sorted:
        raise ValueError(
            "`sorted=True` optimization does not work when v is NaN and a.dtype==object"
        )
    for i, ai in enumerate(a):
        if isinstance(ai, (float, np.datetime64)) and np.isnan(ai):
            return i
    return -1


def _float_find_sorted(a, v, rtol=1e-05, atol=1e-08):
    """
    a ndarray of ints or floats
    v float
    """
    delta = atol + rtol * abs(v)
    minv = v - delta
    maxv = v + delta
    n = a.shape[0]
    i = np.searchsorted(a, minv)
    if i == n or a[i] > maxv:
        return -1
    else:
        return i


def _float_find_unsorted(a, v, rtol=1e-05, atol=1e-08):
    """
    a ndarray of ints or floats
    v float
    """
    indices = np.where(np.isclose(a, v, rtol=rtol, atol=atol))
    if len(indices[0]):
        if a.ndim == 1:
            return indices[0][0]
        else:
            return next(zip(*indices))
    else:
        return -1


def find(a, v, rtol=1e-05, atol=1e-08, sorted=False, default=-1, raises=False):
    """
    Returns the index of the first element in `a` equal to `v`.
    If either a or v (or both) is of floating type, the parameters
    `atol` (absolute tolerance) and `rtol` (relative tolerance)
    are used for comparison (see `np.isclose()` for details).

    Otherwise, returns the `default` value (-1 by default)
    or raises a `ValueError` if `raises=True`.

    In 2D and above the the values in `a` are always tested and returned in
    row-major, C-style order.

    For example,
    >>> find([3, 1, 4, 1, 5], 4)
    2
    >>> find([1, 2, 3], 7)
    -1
    >>> find([1.1, 1.2, 1.3], 1.2)
    1
    >>> find(np.arange(0, 1, 0.1), 0.3)
    3
    >>> find([[3, 8, 4], [5, 2, 7]], 7)
    (1, 2)
    >>> find([[3, 8, 4], [5, 2, 7]], 9)
    -1
    >>> find([999980., 999990., 1e6], 1e6)
    1
    >>> find([999980., 999990., 1e6], 1e6, rtol=1e-9)
    2
    """
    a = np.asarray(a)

    if sorted and a.ndim != 1:
        raise ValueError(
            f"`sorted=True` optimization only works for 1D arrays, a.ndim={a.ndim}"
        )

    complex_mode = float_mode = datetime_mode = nan_mode = False
    if np.issubdtype(a.dtype, np.complexfloating):
        if not isinstance(v, complex):
            v = complex(v)
        complex_mode = True
    elif isinstance(v, complex):
        complex_mode = True
    elif np.issubdtype(a.dtype, np.floating):
        if not isinstance(v, float):
            v = float(v)
        float_mode = True
    elif np.issubdtype(a.dtype, np.number) and isinstance(v, float):
        float_mode = True
    elif np.issubdtype(a.dtype, np.datetime64):
        if not isinstance(v, np.datetime64):
            raise ValueError(
                f"Incompatible data types of a ({a.dtype}) and v ({type(v)})"
            )
        datetime_mode = True
    elif isinstance(v, (float, np.datetime64)) and np.isnan(v):
        nan_mode = True

    if complex_mode:
        if sorted:
            raise ValueError(
                "`sorted=True` optimization cannot be used with complex numbers"
            )
        elif np.isfinite(v):
            res = _float_find_unsorted(a, v, rtol=rtol, atol=atol)
        else:
            res = _generic_float_find(a, v, sorted=False)
    elif float_mode:
        if np.isfinite(v):
            if sorted:
                res = _float_find_sorted(a, v, rtol=rtol, atol=atol)
            else:
                res = _float_find_unsorted(a, v, rtol=rtol, atol=atol)
        else:
            res = _generic_float_find(a, v, sorted=sorted)
    elif datetime_mode and np.isnat(v):
        res = _generic_float_find(a, v, sorted=sorted)
    elif nan_mode:
        res = _nan_find(a, sorted=sorted)
    else:
        res = _generic_find(a, v, sorted=sorted)

    if res == -1:
        if raises:
            raise ValueError(f"{v} is not in array")
        else:
            return default
    return res


# ____________________________  first_above ________________________________


def first_above(a, v, sorted=False, missing=-1, raises=False):
    """
     Returns the index of the first element in `a` strictly greater than `v`.
     If either a or v (or both) is of floating type, the parameters
     `atol` (absolute tolerance) and `rtol` (relative tolerance)
     are used for comparison (see `np.isclose()` for details).

     In 2D and above the the values in `a` are always tested and returned in
     row-major, C-style order.

     If there is no value in `a` greater than `v`, returns the `default` value
     (-1 by default) or raises a `ValueError` if `raises=True`.

     Parameters
     ----------
     v : scalar
    `a` : 1-D array_like
    `v` : scalar
    `sorted` : use bisection to further accelerate the search. Only works for sorted arrays.
    `missing` : the value to return if no element in `a` is greater than `v`
    `raises` : if `True` return an exception instead of returning anything

     For example,
     >>> first_above([4, 5, 8, 2, 7], 6)
     2
     >>> first_above([[4, 5, 8], [2, 7, 3]], 6)
     (0, 2)
     >>> first_above([5, 6, 7], 9)
     3
    """
    a = np.asarray(a)

    if np.issubdtype(a.dtype, complex) or isinstance(v, complex):
        raise ValueError("Complex numbers are not comparable.")

    if np.issubdtype(a.dtype, bool) or isinstance(v, bool):
        raise ValueError("`bool` type is not supported.")

    if a.ndim != 1:
        raise ValueError(
            f"`a` is expected to be 1-dimensional, got {a.ndim}-dimensional array instead"
        )

    if sorted:
        res = np.searchsorted(a, v, side="right")
        if res == a.shape[0]:
            res = -1
    else:
        indices = np.where(a > v)
        if len(indices[0]):
            res = indices[0][0]
        else:
            res = -1

    if res == -1:
        if raises:
            raise ValueError(f"No values above {v} in the array")
        else:
            return missing
    return res


# ______________________________  first_nonzero ___________________________________


def first_nonzero(a, missing=-1, raises=False):
    """
    Returns the index of the first nonzero element in `a`.

    In 2D and above the the values in `a` are always tested and returned in
    row-major, C-style order.

    For example,
    >>> first_nonzero([0, 0, 7, 0, 5])
    2
    >>> first_nonzero([False, True, False, False, True])
    1
    >>> first_nonzero([[0, 0, 0, 0], [0, 0, 5, 3]])
    (1, 2)
    """
    a = np.asarray(a)

    if a.ndim != 1:
        raise ValueError(
            f"`a` is expected to be 1-dimensional, got {a.ndim}-dimensional array instead"
        )

    indices = np.nonzero(a)
    if len(indices[0]):
        res = indices[0][0]
    else:
        res = -1

    if res == -1:
        if raises:
            raise ValueError("All values in `a` are zeros.")
        else:
            return missing
    return res
