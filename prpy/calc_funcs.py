# -*- coding: utf-8 -*-

"""
Functions to execute phase retrieval.
Currently following functions are implemented:
* Error Reduction (ER)
* Fineup's Hybrid Input-Output (HIO)
* Hybrid Projection-Reflection (HPR)
* Oversampling Smoothness (OSS)
"""

import os
import datetime
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from matplotlib import pyplot as plt
from matplotlib import cm

from .filters import FILTERS
from .fft_funcs import FFT_FUNC, IFFT_FUNC
from .Plan import Plan
from .utils import init_rho as _init_rho
from .utils import calc_r_factor as _calc_r_factor
from .utils import update_mask as _updmask
from .proj_density import projection_er as _projection_er
from .proj_density import projection_hio as _projection_hio
from .proj_density import projection_hpr as _projection_hpr
from .proj_modulus import projection_modulus as _projection_I

def _calcpr_er(F, C_s, plan, D_s=None, rho_0=None, r_factor=None, *args, **kwargs):
    """
    Error reduction algorithm
    Detail: J. R. Fienup, Appl. Opt. 21, 2758 (1982)
    """

    # Initialize
    if rho_0 is None:
        rho_0 = _init_rho(plan.shape)
        rho_0 *= np.max(np.abs(ifft2(F)))
    if r_factor is None:
        r_factor = []
    _F = np.abs(fftshift(F))
    _D_s = D_s
    if D_s is not None:
        _D_s = fftshift(D_s)

    rho_i = 1.*rho_0
    rho_f = np.zeros(F.shape, dtype=np.complex64)

    # Define FFT functions
    func = FFT_FUNC.get(plan.fft_type)
    ifunc = IFFT_FUNC.get(plan.fft_type)

    # Check use of mask and validity of the parameters
    updmask_use = plan.kwargs.get('updmask_use', False)
    updmask_N = plan.kwargs.get('updmask_N')
    if updmask_use is None or type(updmask_use) != bool:
        updmask_use = False
    elif updmask_N is None or type(updmask_N) != int or updmask_N <= 0:
        updmask_use = False
    ratio = plan.kwargs.get('updmask_ratio')
    if ratio is None or ratio <= 0:
        ratio = -1

    # The known error function for Liu's process.
    err = plan.kwargs.get('err')
    if err is not None and type(err) not in [np.ndarray, list]:
        raise TypeError('"err" has an invalid type.')
    intensity = plan.kwargs.get('intensity')
    if intensity is not None and type(intensity) != bool:
        raise TypeError('"intensity" must be boolean.')

    _num = 0 if plan.kwargs.get('num') is None else plan.kwargs.get('num')

    # Main loop
    width = 1.5
    for ii in range(plan.N):
        rho_f = func(rho_i, plan.x_gpu, plan.xf_gpu, plan.cufft_plan) # rho(n) -> G(n)
        r_factor.append(_calc_r_factor(rho_f, _F))
        rho_f = _projection_I(rho_f, _F, plan.f_const, _D_s, err, intensity, **kwargs) # G(n) -> G'(n)
        rho_i = _projection_er(ifunc(rho_f, plan.x_gpu, plan.xf_gpu, plan.cufft_plan), C_s, plan.rho_const) # G'(n) -> rho(n+1)
        if updmask_use:
            if np.mod(ii+1, updmask_N) == 0 and ratio > 0: # Update the mask
                width = width-0.03 if width >= 1.5 else 1.5
                C_s = _updmask(np.abs(rho_i), C_s, plan.rho_filter, width, ratio)
        if plan.kwargs.get('save') is True:
            _savefig(np.abs(rho_i), C_s, rho_f, r_factor, plan.pr_mode, ii+_num)
    plan.set(rho_i, r_factor, C_s)


def _calcpr_hio(F, C_s, plan, D_s=None, rho_0=None, r_factor=None, *args, **kwargs):
    """
    Hybrid input-output algorithm
    Detail: J. R. Fienup, Appl. Opt. 21, 2758 (1982)
    """

    # Initialize
    if rho_0 is None:
        rho_0 = _init_rho(plan.shape)
        rho_0 *= np.max(np.abs(ifft2(F)))
    if r_factor is None:
        r_factor = []
    _F = np.abs(fftshift(F))
    _D_s = D_s
    if D_s is not None:
        _D_s = fftshift(D_s)

    rho_i = 1.*rho_0
    rho_f = np.zeros(F.shape, dtype=np.complex64)
    beta = plan.kwargs.get('beta')
    if beta is None or beta <= 0:
        beta = 0.85

    # Define FFT functions
    func = FFT_FUNC.get(plan.fft_type)
    ifunc = IFFT_FUNC.get(plan.fft_type)

    # Check use of mask and validity of the parameters
    updmask_use = plan.kwargs.get('updmask_use', False)
    updmask_N = plan.kwargs.get('updmask_N')
    if updmask_use is None or type(updmask_use) != bool:
        updmask_use = False
    elif updmask_N is None or type(updmask_N) != int or updmask_N <= 0:
        updmask_use = False
    ratio = plan.kwargs.get('updmask_ratio')
    if ratio is None or ratio <= 0:
        ratio = -1

    # The known error function for Liu's process.
    err = plan.kwargs.get('err')
    if err is not None and type(err) not in [np.ndarray, list]:
        raise TypeError('"err" has an invalid type.')
    intensity = plan.kwargs.get('intensity')
    if intensity is not None and type(intensity) != bool:
        raise TypeError('"intensity" must be boolean.')

    _num = 0 if plan.kwargs.get('num') is None else plan.kwargs.get('num')

    # Main loop
    width = 1.5
    for ii in range(plan.N):
        rho_f = func(rho_i, plan.x_gpu, plan.xf_gpu, plan.cufft_plan) # rho(n) -> G(n)
        r_factor.append(_calc_r_factor(rho_f, _F))
        rho_f = _projection_I(rho_f, _F, plan.f_const, _D_s, err, intensity, **kwargs) # G(n) -> G'(n)
        rho_i = _projection_hio(ifunc(rho_f, plan.x_gpu, plan.xf_gpu, plan.cufft_plan), C_s, plan.rho_const, rho_i, beta) # G'(n) -> rho(n+1)
        if updmask_use:
            if np.mod(ii+1, updmask_N) == 0 and ratio > 0: # Update the mask
                width = max(width-0.03, 1.5)
                C_s = _updmask(np.abs(rho_i), C_s, plan.rho_filter, width, ratio)
        if plan.kwargs.get('save') is True:
            _savefig(np.abs(rho_i), C_s, rho_f, r_factor, plan.pr_mode, ii+_num)
    plan.set(rho_i, r_factor, C_s)

def _calcpr_hpr(F, C_s, plan, D_s=None, rho_0=None, r_factor=None, *args, **kwargs):
    """
    Hybrid projection-reflection algorithm
    Detail: H. H. Bauschke, P. L. Combettes and D. R. Luke, J. Opt. Soc. Am. A 20, 1025 (2003)
    """
    # Initialize
    if rho_0 is None:
        rho_0 = _init_rho(plan.shape)
        rho_0 *= np.max(np.abs(ifft2(F)))
    if r_factor is None:
        r_factor = []
    _F = np.abs(fftshift(F))
    _D_s = D_s
    if D_s is not None:
        _D_s = fftshift(D_s)

    rho_i = 1.*rho_0
    rho_f = np.zeros(F.shape, dtype=np.complex64)
    beta = plan.kwargs.get('beta')
    if beta is None or beta <= 0:
        beta = 0.85

    # Define FFT functions
    func = FFT_FUNC.get(plan.fft_type)
    ifunc = IFFT_FUNC.get(plan.fft_type)

    # Check use of mask and validity of the parameters
    updmask_use = plan.kwargs.get('updmask_use', False)
    updmask_N = plan.kwargs.get('updmask_N')
    if updmask_use is None or type(updmask_use) != bool:
        updmask_use = False
    elif updmask_N is None or type(updmask_N) != int or updmask_N <= 0:
        updmask_use = False
    ratio = plan.kwargs.get('updmask_ratio')
    if ratio is None or ratio <= 0:
        ratio = -1

    # The known error function for Liu's process.
    err = plan.kwargs.get('err')
    if err is not None and type(err) not in [np.ndarray, list]:
        raise TypeError('"err" has an invalid type.')
    intensity = plan.kwargs.get('intensity')
    if intensity is not None and type(intensity) != bool:
        raise TypeError('"intensity" must be boolean.')

    _num = 0 if plan.kwargs.get('num') is None else plan.kwargs.get('num')

    # Main loop
    width = 1.5
    for ii in range(plan.N):
        rho_f = func(rho_i, plan.x_gpu, plan.xf_gpu, plan.cufft_plan) # rho(n) -> G(n)
        r_factor.append(_calc_r_factor(rho_f, _F))
        rho_f = _projection_I(rho_f, _F, plan.f_const, _D_s, err, intensity, **kwargs) # G(n) -> G'(n)
        rho_i = _projection_hpr(ifunc(rho_f, plan.x_gpu, plan.xf_gpu, plan.cufft_plan), C_s, plan.rho_const, rho_i, beta) # G'(n) -> rho(n+1)
        if updmask_use:
            if np.mod(ii+1, updmask_N) == 0 and ratio > 0: # Update the mask
                width = width-0.03 if width >= 1.5 else 1.5
                C_s = _updmask(np.abs(rho_i), C_s, plan.rho_filter, width, ratio)
        if plan.kwargs.get('save') is True:
            _savefig(np.abs(rho_i), C_s, rho_f, r_factor, plan.pr_mode, ii+_num)
    plan.set(rho_i, r_factor, C_s)

def _calcpr_oss(F, C_s, plan, D_s=None, rho_0=None, r_factor=None, *args, **kwargs):
    """
    OverSampling smoothness algorithm
    Detail: J. A. Rodriguez et al., J. Appl. Cryst. 46, 12 (2013)
    """
    # Initialize
    if rho_0 is None:
        rho_0 = _init_rho(plan.shape)
        rho_0 *= np.max(np.abs(ifft2(F)))
    if r_factor is None:
        r_factor = []
    _F = np.abs(fftshift(F))
    _D_s = D_s
    if D_s is not None:
        _D_s = fftshift(D_s)

    rho_i = 1.*rho_0
    rho_f = np.zeros(F.shape, dtype=np.complex64)
    beta = plan.kwargs.get('beta')
    if beta is None or beta <= 0:
        beta = 0.85

    # Define FFT functions
    func = FFT_FUNC.get(plan.fft_type)
    ifunc = IFFT_FUNC.get(plan.fft_type)

    # Check use of mask and validity of the parameters
    updmask_use = plan.kwargs.get('updmask_use', False)
    updmask_N = plan.kwargs.get('updmask_N')
    if updmask_use is None or type(updmask_use) != bool:
        updmask_use = False
    elif updmask_N is None or type(updmask_N) != int or updmask_N <= 0:
        updmask_use = False
    ratio = plan.kwargs.get('updmask_ratio')
    if ratio is None or ratio <= 0:
        ratio = -1

    # The known error function for Liu's process.
    err = plan.kwargs.get('err')
    if err is not None and type(err) not in [np.ndarray, list]:
        raise TypeError('"err" has an invalid type.')
    intensity = plan.kwargs.get('intensity')
    if intensity is not None and type(intensity) != bool:
        raise TypeError('"intensity" must be boolean.')

    alpha = plan.N
    d_alpha = - (plan.N-1./plan.N)/9.
    updoss_N = int(plan.N/10)
    _qx = np.linspace(-1., 1.-2./plan.shape[0], plan.shape[0])
    _qy = np.linspace(-1., 1.-2./plan.shape[1], plan.shape[1])
    _qxx, _qyy = np.meshgrid(_qx, _qy)
    _qrr = np.sqrt(_qxx**2 + _qyy**2)
    del _qxx, _qyy, _qx, _qy

    _num = 0 if plan.kwargs.get('num') is None else plan.kwargs.get('num')

    # Main loop
    width = 1.5
    _W = np.exp(-0.5*(_qrr / alpha)**2)
    for ii in range(plan.N):
        rho_f = func(rho_i, plan.x_gpu, plan.xf_gpu, plan.cufft_plan) # rho(n) -> G(n)
        r_factor.append(_calc_r_factor(rho_f, _F))
        rho_f = _projection_I(rho_f, _F, plan.f_const, _D_s, err, intensity, **kwargs) # G(n) -> G'(n)
        rho_i = _projection_hio(ifunc(rho_f, plan.x_gpu, plan.xf_gpu, plan.cufft_plan), C_s, plan.rho_const, rho_i, beta) # G'(n) -> rho(n+1)
        buff = func(rho_i, plan.x_gpu, plan.xf_gpu, plan.cufft_plan)
        buff = np.real(ifunc(buff*_W, plan.x_gpu, plan.xf_gpu, plan.cufft_plan)) # take real part of density
        rho_i = rho_i*C_s + buff*(1-C_s)
        if np.mod(ii+1, updoss_N) == 0:
            alpha -= d_alpha
            _W = np.exp(-0.5*(_qrr / alpha)**2)
        if updmask_use:
            if np.mod(ii+1, updmask_N) == 0 and ratio > 0: # Update the mask
                width = width-0.03 if width >= 1.5 else 1.5
                C_s = _updmask(np.abs(rho_i), C_s, plan.rho_filter, width, ratio)
        if plan.kwargs.get('save') is True:
            _savefig(np.abs(rho_i), C_s, rho_f, r_factor, plan.pr_mode, ii+_num)

    plan.set(rho_i, r_factor, C_s)

CALC_PR = {"ER": _calcpr_er, "HIO": _calcpr_hio, "HPR": _calcpr_hpr, "OSS": _calcpr_oss}

def _calc(F, C_s, plans, D_s=None, rho_0=None, r_factor=None, **kwargs):
    pass

def calc(F, C_s, plans, D_s=None, rho_0=None, r_factor=None, **kwargs):
    """
    Main function to calculate phase retrieval
    < Input parameters >
        F            : Fourier modulus (numpy.2darray)
        C_s          : (Initial) support in real space (numpy.2darray)
        plans        : Plan / list of Plans object (Plan)
        D_s          : (Initial) support in frequency space (numpy.2darray)

    < Output >
        None (results are saved in plans.)
    """

    """--- Check the validity ---"""
    def validity(F, C_s, plan, D_s=None):
        if type(F) is not np.ndarray:
            raise TypeError('The type of "F" must be numpy.ndarray .')
        if F.shape != plan.shape:
            raise ValueError('The size of "F" mismatches that in the plan.')
        if type(C_s) is not np.ndarray:
            raise TypeError('The type of "C_s" must be numpy.ndarray .')
        if C_s.shape != plan.shape:
            raise ValueError('The size of "C_s" mismatches that in the plan.')
        if D_s is not None:
            if type(D_s) is not np.ndarray:
                raise TypeError('The type of "D_s" must be numpy.ndarray .')
            if D_s.shape != plan.shape:
                raise ValueError('The size of "D_s" mismatches that in the plan.')

    if type(plans) == Plan:
        validity(F, C_s, plans, D_s)
        func = CALC_PR.get(plans.pr_mode)
        func(F, C_s, plans, D_s, rho_0, r_factor, **kwargs)
    elif type(plans) == list:
        _num=0
        for plan in plans:
            validity(F, C_s, plan, D_s)
            plan.kwargs['num'] = _num+0
            _num += plan.N
            func = CALC_PR.get(plan.pr_mode)
            func(F, C_s, plan, D_s, rho_0, r_factor, **kwargs)
            rho_0, r_factor = plan.get()
            C_s = 1.*plan.C_s


def _savefig(rho, C_s, F, r_factor, pr_mode, num):
    plt.figure(100, figsize=(18, 12), dpi=100)
    plt.clf()
    plt.suptitle('Mode: {0}, iter={1:d}'.format(pr_mode, num+1), fontsize=15)

    ind = np.where(rho == np.max(rho))
    rho = np.roll(rho, rho.shape[0]//2-ind[0], axis=0)
    rho = np.roll(rho, rho.shape[1]//2-ind[1], axis=1)
    plt.subplot(232)
    # plt.imshow(255-rho, cmap=cm.gray)
    plt.imshow(rho)
    # plt.ylim(ind[0]-50,ind[0]+50)
    # plt.xlim(ind[1]-50,ind[1]+50)
    plt.title('density', fontsize=15)
    plt.xlabel('Pixel', fontsize=15)
    plt.ylabel('Pixel', fontsize=15)
    plt.colorbar()

    plt.subplot(233)
    plt.imshow(C_s)
    # plt.ylim(ind[0]-50,ind[0]+50)
    # plt.xlim(ind[1]-50,ind[1]+50)
    plt.title('real constraint area', fontsize=15)
    plt.xlabel('Pixel', fontsize=15)
    plt.ylabel('Pixel', fontsize=15)

    plt.subplot(234)
    plt.imshow(np.abs(fftshift(F)))
    # plt.xlim(F.shape[0]//2-50,F.shape[0]//2+50)
    # plt.ylim(F.shape[1]//2-50,F.shape[1]//2+50)
    plt.title('Amplitude', fontsize=15)
    plt.xlabel('Pixel', fontsize=15)
    plt.ylabel('Pixel', fontsize=15)
    plt.clim(0, 50000)

    plt.subplot(235)
    plt.imshow(fftshift(np.angle(F)))
    # plt.xlim(F.shape[0]//2-50,F.shape[0]//2+50)
    # plt.ylim(F.shape[1]//2-50,F.shape[1]//2+50)
    plt.title('Phase', fontsize=15)
    plt.xlabel('Pixel', fontsize=15)
    plt.ylabel('Pixel', fontsize=15)
    plt.colorbar()
    plt.clim(-np.pi, np.pi)

    plt.subplot(236)
    plt.plot(range(len(r_factor)), r_factor)
    plt.yscale('log')
    plt.title('R factor')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('R factor', fontsize=15)

    plt.savefig('./images/results_{0:04d}.png'.format(num))
