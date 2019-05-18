from distutils.core import setup
from Cython.Build import cythonize
#import os

#os.environ['CFLAGS'] = '-O3 -Wall'
setup(name='DTW app', ext_modules=cythonize("dtw.pyx"), zip_safe=False)
