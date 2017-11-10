# -*- coding: utf-8 -*-

"""
utils for calculating phase retrieval
"""

import numpy as np
import scipy.interpolate as interp
from .filters import FILTERS, gaussian_filter

def init_rho(shape):
    """ Initialize desity map """
    return np.exp(2*np.pi*1j*np.random.rand(shape[0], shape[1]))

def calc_r_factor(F_cal, F_obs):
    """
    Calculate R-factor.
    $ R = sum(||F_{cal}|-|F_{obs}||)/sum(|F_{obs}|) $
    """
    return np.sum(np.abs(np.abs(F_cal)-np.abs(F_obs)))/np.sum(np.abs(F_obs))

def profile2map(qr, profile, qrr):
    """ Convert a radial profile to a 2D isotropic scattering image """
    f = interp.interp1d(qr, profile, bounds_error=False, fill_value="extrapolate")
    buff = f(qrr.flatten())
    return buff.reshape(qrr.shape)

def update_mask(rho, C_s, filter_type, width, ratio):
    """ Update a support in real space for wrap-shrink algorithm """
    _filter = gaussian_filter if FILTERS.get(filter_type) is None else FILTERS.get(filter_type)
    rho_auto = _filter(rho, width)
    C_s = rho_auto >= ratio*np.max(rho_auto)
    return C_s
