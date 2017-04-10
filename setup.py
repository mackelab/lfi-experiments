from setuptools import setup, find_packages

setup(
    name='lfmods',
    version='0.0.3.dev0',
    description='Applications of likelihoodfree to different problems',
    url='https://github.com/mackelab/likelihoodfree-applications',
    install_requires=['allensdk', 'likelihoodfree', 'h5py', 'nbparameterise'],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    }
)
