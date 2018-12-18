# -*- coding: utf-8 -*-

"""
utils for calculating phase retrieval
"""

import numpy as np
import scipy.interpolate as interp
from .filters import FILTERS, gaussian_filter

def init_rho(shape):
    """init_rho(shape) -> numpy.2darray
    initialize desity map with random distribution
    """
    return np.exp(2*np.pi*1j*np.random.rand(shape[0], shape[1]))

def calc_r_factor(F_cal, F_obs):
    """calc_r_factor(F_cal, F_obs) -> float
    calculate R-factor by using the following formula:
    $ R = sum(||F_{cal}|-|F_{obs}||)/sum(|F_{obs}|) $

    Parameters
    ----------
    F_cal : numpy.2darray
        falculated data
    F_obs : numpy.2darray
        observed data
    
    Returns
    -------
    R-factor (float)
    """
    return np.sum(np.abs(np.abs(F_cal)-np.abs(F_obs)))/np.sum(np.abs(F_obs))

def profile2map(qr, profile, qrr):
    """profile2map(qr, profile, qrr) -> numpy.2darray
    convert a radial profile to a 2D isotropic scattering image
    
    Parameters
    ----------
    qr      : array-like
        axis data
    profile : array-like
        profile data
    qrr     : numpy.2darray
        qr map data
    
    Returns
    -------
    converted map (numpy.2darray)
    """
    f = interp.interp1d(qr, profile, bounds_error=False, fill_value="extrapolate")
    buff = f(qrr.flatten())
    return buff.reshape(qrr.shape)

def update_mask(rho, C_s, filter_type, width, ratio):
    """update_mask(rho, C_s, filter_type, width, ratio) -> numpy.2darray
    update a support in real space for wrap-shrink algorithm
    
    Parameters
    ----------
    rho         : numpy.2darray
        density map
    C_s         : numpy.2darray
        previous support for density map
    filter_type : str
        type of filter
    width       : float
        width for filter
    ratio       : float in (0, 1)
        ratio where the newer support is given 

    Returns
    -------
    C_s : numpy.2darray
        support for density map
    """
    _filter = gaussian_filter if FILTERS.get(filter_type) is None else FILTERS.get(filter_type)
    rho_auto = _filter(rho, width)
    C_s = rho_auto >= ratio*np.max(rho_auto)
    return C_s
