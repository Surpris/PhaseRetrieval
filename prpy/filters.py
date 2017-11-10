# -*- coding: utf-8 -*-
"""
filters
"""

from cv2 import GaussianBlur as cvGB

def gaussian_filter(F, sigma):
    """ Gaussian Filter """
    neighbours = 2*(int(2.884402748387961466*sigma-0.5) // 2) + 1
    return cvGB(F, (neighbours, neighbours), sigma)

FILTER_TYPES = ['gaussian']
FILTERS = dict(gaussian=gaussian_filter)
