from distutils.core import setup
from Cython.Build import cythonize

setup(name='DTW app', ext_modules=cythonize("dtw.pyx"), zip_safe=False)
