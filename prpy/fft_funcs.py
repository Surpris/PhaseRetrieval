# -*- coding: utf-8 -*-

"""
FFT functions
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import multiprocessing
import importlib

spam_spec = importlib.util.find_spec("pyfftw")
FOUND_FFTW = spam_spec is not None
if FOUND_FFTW is True:
    import pyfftw

spam_spec = importlib.util.find_spec("pycuda")
FOUND_PYCUDA = spam_spec is not None
spam_spec = importlib.util.find_spec("skcuda")
FOUND_SKCUDA = spam_spec is not None
if FOUND_PYCUDA is True and FOUND_SKCUDA is True:
    FOUND_CUFFT = True
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cu_fft
    import pycuda.autoinit
else:
    FOUND_CUFFT = False


FFT_TYPES = ['numpy', 'fftw', 'cufft']

""" Numpy """
def _fft2(F, *args, **kwargs):
    return fft2(F)

def _ifft2(F, *args, **kwargs):
    return ifft2(F)

""" FFTW """
def _fftw(F, *args, **kwargs):
    pyfftw.forget_wisdom()
    func = pyfftw.builders.fft2(F, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
    return func()

def _ifftw(F, *args, **kwargs):
    pyfftw.forget_wisdom()
    ifunc = pyfftw.builders.ifft2(F, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
    return ifunc()

""" cuFFT """
def _cufft(F, x_gpu, xf_gpu, plan):
    x_gpu.set(F.astype(np.complex64))
    cu_fft.fft(x_gpu, xf_gpu, plan)
    return xf_gpu.get()

def _cuifft(F, x_gpu, xf_gpu, plan):
    x_gpu.set(F.astype(np.complex64))
    cu_fft.ifft(x_gpu, xf_gpu, plan, True)
    return xf_gpu.get()

def _fftfuncs(fft_type):
    if fft_type == 'numpy':
        return _fft2, _ifft2
    elif fft_type == 'fftw':
        return _fftw, _ifftw
    elif fft_type == 'cufft':
        return _cufft, _cuifft

FFT_FUNC = dict(numpy=_fft2, fftw=_fftw, cufft=_cufft)
IFFT_FUNC = dict(numpy=_ifft2, fftw=_ifftw, cufft=_cuifft)
