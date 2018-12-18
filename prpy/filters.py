# -*- coding: utf-8 -*-
"""
filters
"""

from cv2 import GaussianBlur as cvGB

def gaussian_filter(F, sigma):
    """gaussian_filter(F, sigma) -> numpy.2darray
    apply Gaussian Filter

    Parameters
    ----------
    F     : numpy.2darray
        target array
    sigma : float
        sigma for Gaussian filter
    
    Returns
    -------
    filtered array
    """
    neighbours = 2*(int(2.884402748387961466*sigma-0.5) // 2) + 1
    return cvGB(F, (neighbours, neighbours), sigma)

FILTER_TYPES = ['gaussian']
FILTERS = dict(gaussian=gaussian_filter)
