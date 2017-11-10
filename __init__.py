# -*- coding: utf-8 -*-

try:
    __PHASERETRIEVAL_SETUP__
except NameError:
    __PHASERETRIEVAL_SETUP__ = False

if __PHASERETRIEVAL_SETUP__:
    import sys as _sys
    _sys.stderr.write('Running from particle source directory.\n')
    del _sys
else:
    from .pypr import *
