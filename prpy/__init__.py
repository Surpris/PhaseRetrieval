# -*- coding: utf-8 -*-

"""
pypr module

This module offers functions for phase retrieval.

Example of usage
----------------
import matplotlib.pyplot as plt
import numpy as np

# import sys
# sys.path.append('path/to/folder/including/this/module')
import prpy

src_path = 'a/b/c.npy' # target modulus
src = np.load(src_path)

C_s_path = 'd/e/f.npy' # support
C_s = np.load(C_s_path)

params = dict(updmask_use=False, save=False, num=0, f_const="free")

plans = []
plans.append(prpy.Plan(src.shape, 'ER', 300, **params))

st = time.time()
prpy.calc(src, C_s, plans)
ft = time.time()
print("Elapsed time: {0:.2f} sec.".format(ft-st))

rho_i, r_factor = plans[len(plans)-1].get()

plt.figure(figsize=(6,5), dpi=100)
plt.subplot(111)
plt.imshow(np.abs(rho_i))
plt.colorbar()

#--- End of this example ---
"""

from .calc_funcs import calc
from .fft_funcs import FFT_FUNC, IFFT_FUNC, FFT_TYPES
from .filters import FILTERS, FILTER_TYPES
from .Plan import Plan, PR_MODES
from .proj_density import projection_er, projection_hio, projection_hpr, RHO_CONSTS
from .proj_modulus import projection_modulus, F_CONSTS
from .utils import init_rho, calc_r_factor, profile2map
