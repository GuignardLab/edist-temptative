#!/usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize

setup(name='edist app', ext_modules=cythonize("edist/*.pyx"), script_args = ['build_ext', '--inplace'])
