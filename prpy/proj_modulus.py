# -*- coding: utf-8 -*-

"""
Constraints in frequency space
"""

import numpy as np
from numpy.fft import fftshift

F_CONSTS = ['normal', 'free', 'nishino', 'complex', 'F(qr)']

def projection_modulus(G, F, f_const, D_s=None, err=None, intensity=False, **kwargs):
    """ Constraints in frequency space.

    < Input parameters >
        G: estimated Fourier modulus
        F: measured Fourier amplitude
        f_const: type of constraint to Fourier modulus
            normal: normalization only
            free: free development out of D_s
            nishino: normalization method proposed by Prof. Nishino
                See "Y. Nishino et al., Phys. Rev. B 68, 220101 (2003)" for more details.
        D_s: constrained region where noise exists.
        err: the known error function (=r(u)) or a list with [lower, upper]
            See "G. Liu, SignalProcessing, 21, 339 (1990)" for more details.
    < Output >
        Constrained Fourier modulus
    """
    buff = F
    if err is not None: # Error constraint: r'' -> r'
        r_upper = max(err)
        r_lower = min(err)
        if intensity is False:
            r = buff - np.abs(G)
            for row in r:
                ind = np.where(row >=r_upper)
                row[ind] = r_upper
                ind = np.where(row <=r_lower)
                row[ind] = r_lower
            buff = buff - r
        else:
            r = buff**2 - np.abs(G)**2
            for row in r:
                ind = np.where(row >=r_upper)
                row[ind] = r_upper
                ind = np.where(row <=r_lower)
                row[ind] = r_lower
            buff = np.sqrt(buff**2 - r)

    if D_s is None or f_const=='normal':
        return np.exp(1j*np.angle(G))*buff
    elif D_s is not None and f_const=='free': # free development
        return G*D_s + np.exp(1j*np.angle(G))*buff*(1-D_s)
    elif D_s is not None and f_const=='nishino': # modified support proposed by Y. Nishino
        return np.exp(1j*np.angle(G))*buff
    elif D_s is not None and f_const == "F(qr)": # F(qr) support
        F_qr = fftshift(kwargs.get("F(qr)"))
        return F_qr*D_s + np.exp(1j*np.angle(G))*buff*(1-D_s)
