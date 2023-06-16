# numpy-illustrated

[![pypi](https://img.shields.io/pypi/v/numpy-illustrated.svg)](https://pypi.python.org/pypi/numpy-illustrated)
[![python](https://img.shields.io/pypi/pyversions/numpy-illustrated.svg)](https://pypi.org/project/numpy-illustrated/)
![pytest](https://github.com/axil/numpy-illustrated/actions/workflows/pytest.yml/badge.svg)
![Coverage Badge](https://github.com/axil/numpy-illustrated/raw/master/img/coverage.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/pypi/l/numpy-illustrated)](https://pypi.org/project/numpy-illustrated/)

This repo contains code for a number of helper functions mentioned in the [NumPy Illustrated](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d?sk=57b908a77aa44075a49293fa1631dd9b) guide.

## Installation: 

    pip install numpy-illustrated

## Contents

Three search functions that return immediately after finding the requested value resulting in a 1000x and more speedup for huge arrays:

  - `find`
  - `first_above`
  - `first_nonzero`
    
For better portability, in this library only the pure python/numpy implementation is provided (no speedup).
The actual cython accelerated code is packaged separately in a library called `ndfind`.
If this library is installed with `pip install ndfind` (binaries are provided for python 3.8 .. 3.11 under 
Windows, Linux and MacOS), the faster versions of the functions are used when calling `npi.find`, etc.

If either the array or the value to be found is of floating type, the floating point comparison with relative 
and absolute tolerances is used.

The next four functions act just like `np.argmin`, `np.argmax`, etc., but return a tuple rather than a scalar 
in 2D and above:  

 - `argmin`  
 - `argmax`
 - `nanargmin`  
 - `nanargmax`

Alternative transpose function that converts 1D (row) vector into 2D column vector and back again:  
  - `T_(a)`

Sort function that is able to sort by selected column(s) in ascending/descending order (like sort_values in Pandas):  
  - `sort`

An inclusive range:  
  - `irange`

An alias to concatenate:
  - `concat`

## Documentation

- `find(a, v, rtol=1e-05, atol=1e-08, sorted=False, default=-1, raises=False)`  

Returns the index of the first element in `a` equal to `v`.
If either a or v (or both) is of floating type, the parameters
`atol` (absolute tolerance) and `rtol` (relative tolerance) 
are used for comparison (see `np.isclose()` for details).

Otherwise, returns the `default` value (-1 by default)
or raises a `ValueError` if `raises=True`.

In 2D and above the the values in `a` are always tested and returned in
row-major, C-style order.

For example,
```python
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
```

- `first_above(a, v, sorted=False, missing=-1, raises=False)`

Returns the index of the first element in `a` strictly greater than `v`.
If either a or v (or both) is of floating type, the parameters
`atol` (absolute tolerance) and `rtol` (relative tolerance) 
are used for comparison (see `np.isclose()` for details).

In 2D and above the the values in `a` are always tested and returned in
row-major, C-style order.

If there is no value in `a` greater than `v`, returns the `default` value 
(-1 by default) or raises a `ValueError` if `raises=True`.

Parameters:  
`a` : 1-D array_like  
`v` : scalar
`sorted` : use bisection to further accelerate the search. Only works for sorted arrays.
`missing` : the value to return if no element in `a` is greater than `v`
`raises` : if `True` return an exception instead of returning anything

For example,
```python
    >>> first_above([4, 5, 8, 2, 7], 6)
    2 
    >>> first_above([[4, 5, 8], [2, 7, 3]], 6)
    (0, 2) 
    >>> first_above([5, 6, 7], 9)
    3 
```

-  `first_nonzero(a, missing=-1, raises=False)`

Returns the index of the first nonzero element in `a`.

In 2D and above the the values in `a` are always tested and returned in
row-major, C-style order.

For example,
```python
    >>> first_nonzero([0, 0, 7, 0, 5])
    2
    >>> first_nonzero([False, True, False, False, True])
    1
    >>> first_nonzero([[0, 0, 0, 0], [0, 0, 5, 3]])
    (1, 2)
```

- `argmin(a)`

Returns the index of the minimum value.
The result is scalar in 1D case and tuple of indices in 2D and above.
If the maximum is encountered several times, returns the first match
in the C order (irrespectively of the order of the array itself).
E.g.:
```python
    >>> argmin([4, 3, 5])
    1
    >>> argmin([[4, 8, 5], [9, 3, 1]])
    (1, 2)
```

- `argmax(a)`

Returns the index of the maximum value.
The result is scalar in 1D case and tuple of indices in 2D and above.
If the maximum is encountered several times, returns the first match
in the C order (irrespectively of the order of the array itself).
E.g.:
```python
    >>> argmax([4, 5, 3])
    1
    >>> argmax([[4, 3, 5], [5, 4, 3]])
    (0, 2)
```

- `nanargmin(a)`

Returns the index of the minimum value.
The result is scalar in 1D case and tuple of indices in 2D and above.
If the maximum is encountered several times, returns the first match
in the C order (irrespectively of the order of the array itself).
E.g.:
```python
    >>> nanargmin([4, 3, nan])
    1
    >>> nanargmin([[4, 8, 5], [9, 3, 1]])
    (1, 2)
```

- `nanargmax(a)`

Returns the index of the maximum value.
The result is scalar in 1D case and tuple of indices in 2D and above.
If the maximum is encountered several times, returns the first match
in the C order (irrespectively of the order of the array itself).
E.g.:
```
    >>> nanargmax([nan,5,3])
    1
    >>> nanargmax([[4,3,5], [5,nan,3]])
    (0, 2)
```

- `T_(x)`

Returns a view of the array with axes transposed:
  - transposes a matrix just like the original T;
  - transposes 1D array to a 2D column-vector and vica versa;
  - transposes (a less commonly used) 2D row-vector to a 2D column-vector;
  - for 3D arrays and above swaps the last two dimensions.
E.g.:
```python
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
```

- `sort(a, by=None, axis=0, ascending=True)`

Rearranges the rows so that the result is sorted by the specified columns
An extension of `sort` that allows sorting by column(s), ascending and descending.

If by is a list [c1, c2, ..., cn], sorts by the column c1, resolving the ties using
the column c2, and so on until cn (just like in pandas). Unlike pandas, the columns
not present in the `by` argument are used for resolving the remaining ties in the
left to right order.

`by=None` is the same as by=[0, 1, 2, ..., a.shape[-1]]

`ascending` can be either be a scalar or a list.

For example:
```python
    >>>  sort([[1, 2, 3],
               [3, 1, 5],
               [1, 0, 6]])
    array([[1, 0, 6],
           [1, 2, 3],
           [3, 1, 5]])
```

- `irange(start, stop, step=1, dtype=None, tol=1e-6)`

Returns an evenly spaced array from start to stop inclusively.
If the range `stop-start` is not evenly divisible by step (=if the calculated number 
of steps is further from the nearest integer than `tol`), raises a ValueError 
exception.

- `concat`

Just a shorter alias to `np.concatenate`

## Testing

Run `pytest` in the project root.
