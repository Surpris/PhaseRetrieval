prpy: Phase Retrieval in Python
=====

(under construction)   
This module aims to calculate phase retrieval (PR) in Python.   

# Requirement
This module has dependencies on the following modules.

* Python (>= 3.4)
    + Probably this module is available on Python 2.7 (not tested).
* NumPy (>= 1.11)
* Matplotlib (>= 1.5)
    + mpl_toolkits.mplot3d
* importlib

If you want to do calculation with `pyfftw` or `pycuda`, the following modules are also required.

* pyfftw (for pyfftw)
* multiprocessing (for pyfftw)
* pycuda (for pycuda)
* skcuda (for pycuda)

# Installation
Currently, installing this module requires the following steps:

1. Download this module as a zip (or tar.gz) file.
1. Unzip the file to to any directory.
1. Run `python setup.py install` at the directory.

If the above procedure does not work, please try the following:

1. Download this module as a zip (or tar.gz) file.
1. Unzip the file to to any directory.
1. Run `python setup.py sdist` at the directory.
    + A package file `prpy-x.x.x.tar.gz` are created in `dist` directory.
1. Move to `dist` directory.
1. Run `pip install ./prpy-x.x.x.tar.gz` at the directory.

# Hierarchical structure of prpy module
```
/prpy
    calc_cuncs.py    : calculation functions
    fft_funcs.py     : FFT wrappers
    filters.py       : filter functions
    Plan.py          : Plan class of PR, like Plan class of pyfftw
    proj_density.py  : projection functions in real space (error reduction, etc.)
    proj_modulus.py  : projection functions in frequency space
    utils.py         :
```

# Projection functions
In real space, following functions are available:

* `projection_er`  : error reduction algorithm
* `projection_hio` : hybrid input-output algorithm
* `projection_hpr` : hybrid projection-reflection algorithm

In frequency space, `projection_modulus` applies constraints to a Fourier modulus.
