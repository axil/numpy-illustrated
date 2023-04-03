import zipfile

import numpy as np
from numpy.compat import (
    asbytes, asstr, asunicode, os_fspath, os_PathLike,
    pickle
    )

def argmin(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim > 1:
        return np.unravel_index(np.argmin(a), a.shape)
    else:
        return np.argmin(a)

def argmax(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim > 1:
        return np.unravel_index(np.argmax(a), a.shape)
    else:
        return np.argmax(a)

def T(x):
    x = np.array(x)
    if x.ndim == 1:
        return x[:, None]
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x[:,0]
        else:
            return x.T
    else:
        return np.swapaxes(x, x.ndim-2, x.ndim-1)


def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.
    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    if not hasattr(file, 'read'):
        file = os_fspath(file)
    import zipfile
    kwargs['allowZip64'] = True
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
        if not hasattr(self.file, 'write'):
            self.file = os_fspath(self.file)
            if not self.file.endswith('.npz'):
                self.file = self.file + '.npz'

        if self.compress:
            compression = zipfile.ZIP_DEFLATED
        else:
            compression = zipfile.ZIP_STORED
            
        self.zipf = zipfile_factory(self.file, mode="w", compression=compression)
        return self

    def write(self, *args, **kwargs):
        namedict = kwargs
        for i, val in enumerate(args, start=self.idx):
            key = 'arr_%d' % i
            if key in namedict.keys():
                raise ValueError(
                    "Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
        self.idx += i+1

        for key, val in namedict.items():
            fname = key + '.npy'
            val = np.asanyarray(val)
            # always force zip64, gh-10776
            with self.zipf.open(fname, 'w', force_zip64=True) as fid:
                np.lib.format.write_array(fid, val,
                                   allow_pickle=self.allow_pickle,
                                   pickle_kwargs=self.pickle_kwargs)
        
    def __exit__(self, *args):
        self.zipf.close()

