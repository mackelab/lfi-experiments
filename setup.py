import numpy as np

from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages

extensions = [Extension('lfmods.hh_bm_cython_comp', ['lfmods/hh_bm_cython_comp.pyx'], include_dirs = [np.get_include()])]

setup(
    name='lfmods',
    version='0.0.5.dev0',
    description='Applications of likelihoodfree to different problems',
    url='https://github.com/mackelab/likelihoodfree-applications',
    install_requires=['allensdk', 'likelihoodfree', 'h5py', 'nbparameterise',
                      'redis', 'rq'],
    ext_modules = cythonize(extensions),
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    }
)
