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
    """Plan class of phase retrieval (PR) algorithm"""

    def __init__(self, shape, pr_mode, N, fft_type='numpy', rho_const='real', f_const='normal',
                      rho_filter="gaussian", f_filter='gaussian', *args, **kwargs):
        """__init__(self, shape, pr_mode, N, fft_type='numpy', rho_const='real', f_const='normal',
                      rho_filter="gaussian", f_filter='gaussian', *args, **kwargs) -> None
        initialize this class
        
        Parameters
        ----------
        shape      : 2-value tuple
            shape of input Fourier modulus
        pr_mode    : str
            mode of PR (in .PR_MODES)
        N          : positive int
            # of iterations
        fft_type   : str
            FFT type (in .fft_funcs.FFT_TYPES)
        rho_const  : str
            Constraint type in real space (in .proj_density.RHO_CONSTS)
        f_const    : str
            Constraint type in reciprocal space (in .proj_modulus.F_CONSTS)
        rho_filter : str
            Filter type to density (in .filters.FILTER_TYPES)
        f_filter   : str
            Filter type to modulus (in .filters.FILTER_TYPES)
        args       : options
        kwargs     : options
            # HIO/HPR
            beta          : float
                Coefficient of HIO/HPR
            # Wrap-shrink algorithm (S. Marchesini et al., Phys. Rev. B 68, 140101 (2003))
            updmask_use   : bool
                Use/unuse of wrap-shrink algorithm
            updmask_N     : positive int
                Uprate frequency
            updmask_ratio : float in (0, 1)
                Threshold in generation of mask
            sigma_start   : float
                Initial sigma of Gaussian filter
            sigma_end     : float
                Terminal sigma of Gaussian filter
            sigma_rate    : float in (0, 1)
                Decrease rate of sigma of Gaussian filter
        """

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
        """get(self) -> numpy.2darray, list
        return rho and R factor
        """
        return self.rho_i.copy(), self.r_factor

    def set(self, rho_i=None, r_factor=None, C_s=None):
        """set(self, rho_i=None, r_factor=None, C_s=None) -> None
        set rho, R factor, and spatial constraint

        Parameters
        ----------
        rho_i    : numpy.2darray
        f_factor : list
        C_s      : numpy.2darray
        """
        self.rho_i= 1.*rho_i if rho_i is not None else rho_i
        self.r_factor = r_factor
        self.C_s = 1.*C_s

    def save(self, filename):
        """save(self, filename) -> None
        save properties

        Parameters
        ----------
        filename : str
            file path to save properties to
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
        """output(self) -> dict
        output properties
        """
        return dict(shape=self.shape, pr_mode=self.pr_mode,
                    N_iter=self.N, fft_type=self.fft_type,
                    rho_const=self.rho_const, f_const=self.f_const,
                    rho_filter=self.rho_filter, f_filter=self.f_filter,
                    rho_i=self.rho_i, C_s=self.C_s,
                    F=fft2(self.rho_i), r_factor=self.r_factor, **self.kwargs)
