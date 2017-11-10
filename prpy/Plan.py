# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft2
import pickle
import importlib

from .fft_funcs import FFT_TYPES, FOUND_FFTW, FOUND_CUFFT
from .filters import FILTER_TYPES
from .proj_density import RHO_CONSTS
from .proj_modulus import F_CONSTS

PR_MODES = ["ER", "HIO", "HPR", "OSS", "Liu"]

class Plan:
    """ Plan class of phase retrieval (PR) algorithm.
    < Input parameters >
        shape       : shape of input Fourier modulus (2-value tuple)
        pr_mode     : mode of PR (string in .PR_MODES)
        N           : # of iterations (unsigned integer)
        fft_type    : FFT type (string in .fft_funcs.FFT_TYPES)
        rho_const   : Constraint type in real space (string in .proj_density.RHO_CONSTS)
        f_const     : Constraint type in frequency space (string in .proj_modulus.F_CONSTS)
        rho_filter  : Filter type to density (string in .filters.FILTER_TYPES)
        f_filter    : Filter type to modulus (string in .filters.FILTER_TYPES)

    Other following parameters can be given in kwargs.
    # HIO/HPR
        beta        : Coefficient of HIO/HPR (float/double)

    # Wrap-shrink algorithm (S. Marchesini et al., Phys. Rev. B 68, 140101 (2003))
        updmask_use   : Use/unuse of wrap-shrink algorithm (boolean)
        updmask_N     : Uprate frequency (uunsigned integer)
        updmask_ratio : Threshold in generation of mask (float/double in (0, 1))
        sigma_start   : Initial sigma of Gaussian filter (float/double)
        sigma_end     : Terminal sigma of Gaussian filter (float/double)
        sigma_rate    : decrease rate of sigma of Gaussian filter (float/double in (0, 1))
    """

    def __init__(self, shape, pr_mode, N, fft_type='numpy', rho_const='real', f_const='normal',
                      rho_filter="gaussian", f_filter='gaussian', *args, **kwargs):
        """ initialization of Plan """

        if np.isscalar(shape):
            self.shape = (shape, )
        elif len(shape) == 2:
            self.shape = shape
        else:
            raise ValueError('Invalid value for the keyword "shape."')

        if pr_mode not in PR_MODES:
            raise ValueError('Invalid value for the keyword "pr_mode."')
        self.pr_mode = pr_mode
        self.N = N

        if fft_type not in FFT_TYPES:
            raise ValueError('Invalid value for the keyword "fft_type."')
        elif fft_type == FFT_TYPES[1] and FOUND_FFTW is False:
            fft_type = 'numpy'
        elif fft_type == FFT_TYPES[2] and FOUND_CUFFT is False:
            fft_type = 'numpy'

        self.fft_type = fft_type
        if self.fft_type == FFT_TYPES[2] and FOUND_CUFFT is True:
            self.x_gpu = gpuarray.empty(self.shape, np.complex64)
            self.xf_gpu = gpuarray.empty(self.shape, np.complex64)
            self.cufft_plan = cu_fft.Plan(self.shape, np.complex64, np.complex64)
        else:
            self.x_gpu = None
            self.xf_gpu = None
            self.cufft_plan = None

        if rho_const not in RHO_CONSTS:
            raise ValueError('Invalid value for the keyword "rho_const."')
        self.rho_const = rho_const

        if f_const not in F_CONSTS:
            raise ValueError('Invalid value for the keyword "f_const."')
        self.f_const = f_const

        if rho_filter is not None:
            if rho_filter not in FILTER_TYPES:
                raise ValueError('Invalid value for the keyword "rho_filter."')
        self.rho_filter = rho_filter

        if f_filter is not None:
            if f_filter not in FILTER_TYPES:
                raise ValueError('Invalid value for the keyword "f_filter."')
        self.f_filter = f_filter

        self.kwargs = kwargs
        # self.set()

    def get(self):
        """ return rho and R factor """
        return 1.*self.rho_i, self.r_factor

    def set(self, rho_i=None, r_factor=None, C_s=None):
        """ set rho, R factor, and spatial constraint """
        self.rho_i= 1.*rho_i if rho_i is not None else rho_i
        self.r_factor = r_factor
        self.C_s = 1.*C_s

    def save(self, filename):
        """
        save properties.
        filename: file path to save properties to
        """
        _out = dict(shape=self.shape, pr_mode=self.pr_mode,
                    N_iter=self.N, fft_type=self.fft_type,
                    rho_const=self.rho_const, f_const=self.f_const,
                    rho_filter=self.rho_filter, f_filter=self.f_filter,
                    rho_i=self.rho_i, C_s=self.C_s,
                    F=fft2(self.rho_i), r_factor=self.r_factor, **self.kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(_out, f)

    def output(self):
        """ output properties """
        return dict(shape=self.shape, pr_mode=self.pr_mode,
                    N_iter=self.N, fft_type=self.fft_type,
                    rho_const=self.rho_const, f_const=self.f_const,
                    rho_filter=self.rho_filter, f_filter=self.f_filter,
                    rho_i=self.rho_i, C_s=self.C_s,
                    F=fft2(self.rho_i), r_factor=self.r_factor, **self.kwargs)
