#!python
#cython: language_level=3
"""
Implements the dynamic time warping distance and its backtracing in cython.

Copyright (C) 2019
Benjamin Paaßen
AG Machine Learning
Bielefeld University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt
cimport cython
from edist.alignment import Alignment

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def dtw(x, y, delta):
    """ Computes the dynamic time warping distance between the input sequence
    x and the input sequence y, given the element-wise distance function delta.

    Args:
    x:     a sequence of objects.
    y:     another sequence of objects.
    delta: a function that takes an element of x as first and an element of y
           as second input and returns the distance between them.

    Returns: the dynamic time warping distance between x and y according to
             delta.
    """
    cdef int m = len(x)
    cdef int n = len(y)
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    # First, compute all pairwise replacements
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i,j] = delta(x[i], y[j])

    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_numeric(double[:] x, double[:] y):
    """ Computes the dynamic time warping distance between two input arrays x
    and y, using the absolute value as element-wise distance measure.

    Args:
    x:     an array of doubles.
    y:     another array of doubles.

    Returns: the dynamic time warping distance between x and y.
    """
    cdef int m = len(x)
    cdef int n = len(y)
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    # First, compute all pairwise replacements
    # using OMP parallelization
    cdef int i
    cdef int j
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    for i in prange(m, nogil=True):
        for j in prange(n):
            if(x[i] > y[j]):
                Delta_view[i,j] = x[i] - y[j]
            else:
                Delta_view[i,j] = y[j] - x[i]
    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_manhattan(double[:,:] x, double[:,:] y):
    """ Computes the multivariate dynamic time warping distance between two
    input arrays x and y, using the Manhattan distance as element-wise
    distance measure.

    Args:
    x:     an array of doubles.
    y:     another array of doubles.

    Returns: the dynamic time warping distance between x and y.
    """
    cdef int m = x.shape[0]
    cdef int n = y.shape[0]
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    cdef int K = x.shape[1]
    if(y.shape[1] != K):
        raise ValueError('x and y do not have the same dimensionality (%d versus %d)' % (x.shape[1], y.shape[1]))
    # First, compute all pairwise replacements
    # using OMP parallelization
    cdef int i
    cdef int j
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    cdef int k
    cdef double diff
    for i in prange(m, nogil=True):
        for j in prange(n):
            for k in prange(K):
                diff = x[i,k] - y[j,k]
                if(diff < 0):
                    Delta_view[i, j] -= diff
                else:
                    Delta_view[i, j] += diff
    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_euclidean(double[:,:] x, double[:,:] y):
    """ Computes the multivariate dynamic time warping distance between two
    input arrays x and y, using the Euclidean distance as element-wise
    distance measure.

    Args:
    x:     an array of doubles.
    y:     another array of doubles.

    Returns: the dynamic time warping distance between x and y.
    """
    cdef int m = x.shape[0]
    cdef int n = y.shape[0]
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    cdef int K = x.shape[1]
    if(y.shape[1] != K):
        raise ValueError('x and y do not have the same dimensionality (%d versus %d)' % (x.shape[1], y.shape[1]))
    # First, compute all pairwise replacements
    # using OMP parallelization
    cdef int i
    cdef int j
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    cdef int k
    cdef double diff
    for i in prange(m, nogil=True):
        for j in prange(n):
            for k in prange(K):
                diff = x[i,k] - y[j,k]
                Delta_view[i, j] += diff * diff
            Delta_view[i, j] = sqrt(Delta_view[i, j])
    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_string(str x, str y):
    """ Computes the multivariate dynamic time warping distance between two
    input strings x and y, using the Kronecker distance as element-wise
    distance measure.

    Args:
    x:     a string.
    y:     another string.

    Returns: the dynamic time warping distance between x and y.
    """
    cdef int m = len(x)
    cdef int n = len(y)
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    # First, compute all pairwise replacements
    cdef int i
    cdef int j
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    for i in range(m):
        for j in range(n):
            if(x[i] == y[j]):
                Delta_view[i, j] = 0.
            else:
                Delta_view[i, j] = 1.
    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
cdef void dtw_c(const double[:,:] Delta, double[:,:] D) nogil:
    """ Computes the dynamic time warping distance between two input sequences
    with pairwise element distances Delta and an (empty) dynamic programming
    matrix D.

    Args:
    Delta:   a m x n matrix containing the pairwise element distances.
    D:       another m x n matrix to which the output will be written.
             The dynamic time warping distance will be in cell [0, 0]
             after the computation is finished.
    """
    cdef int i
    cdef int j
    # initialize last entry
    D[-1, -1] = Delta[-1, -1]
    # compute last column
    for i in range(D.shape[0]-2,-1,-1):
        D[i,-1] = Delta[i,-1] + D[i+1,-1]
    # compute last row
    for j in range(D.shape[1]-2,-1,-1):
        D[-1,j] = Delta[-1,j] + D[-1,j+1]
    # compute remaining matrix
    for i in range(D.shape[0]-2,-1,-1):
        for j in range(D.shape[1]-2,-1,-1):
            D[i,j] = Delta[i,j] + min3(D[i+1,j+1], D[i,j+1], D[i+1,j])

cdef double min3(double a, double b, double c) nogil:
    """ Computes the minimum of three numbers.

    Args:
    a: a number
    b: another number
    c: yet another number

    Returns: min({a, b, c})
    """
    if(a < b):
        if(a < c):
            return a
        else:
            return c
    else:
        if(b < c):
            return b
        else:
            return c

####### BACKTRACING FUNCTIONS #######

cdef double _BACKTRACE_TOL = 1E-5

def dtw_backtrace(x, y, delta):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the element-wise distance function delta. This mechanism
    is deterministic and will always prefer replacements over other options.

    Args:
    x:     a sequence of objects.
    y:     another sequence of objects.
    delta: a function that takes an element of x as first and an element of y
           as second input and returns the distance between them.

    Returns: a co-optimal alignment.Alignment between x and y.
    """
    cdef int m = len(x)
    cdef int n = len(y)
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    # First, compute all pairwise replacements
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i,j] = delta(x[i], y[j])

    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)

    cdef double[:,:] D_view = D
    # Finally, compute the backtrace
    i = 0
    j = 0
    alignment = Alignment()
    while(i < m - 1 and j < n - 1):
        alignment.append_tuple(i, j)
        # check which alignment option is co-optimal
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j+1]):
            # replacement is co-optimal
            i += 1
            j += 1
            continue
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j]):
            # copying y[j] is co-optimal
            i += 1
            continue
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i,j+1]):
            # copying x[i] is co-optimal
            j += 1
            continue
        # if we got here, nothing is co-optimal, which is an error
        raise ValueError('Internal error: No option is co-optimal.')
    while(i < m - 1):
        alignment.append_tuple(i, j)
        i += 1
    while(j < n - 1):
        alignment.append_tuple(i, j)
        j += 1
    alignment.append_tuple(m-1, n-1)
    return alignment

import random

def dtw_backtrace_stochastic(x, y, delta):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the element-wise distance function delta. This mechanism
    is stochastic and will return a random co-optimal alignment.

    Args:
    x:     a sequence of objects.
    y:     another sequence of objects.
    delta: a function that takes an element of x as first and an element of y
           as second input and returns the distance between them.

    Returns: a co-optimal alignment.Alignment between x and y.
    """
    cdef int m = len(x)
    cdef int n = len(y)
    if(m < 1 or n < 1):
        raise ValueError('Dynamic time warping can not handle empty input sequences!')
    # First, compute all pairwise replacements
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i,j] = delta(x[i], y[j])

    # Then, compute the dynamic time warping
    # distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)

    cdef double[:,:] D_view = D
    # Finally, compute the backtrace
    cdef int r
    i = 0
    j = 0
    alignment = Alignment()
    while(i < m - 1 and j < n - 1):
        alignment.append_tuple(i, j)
        # check which alignment options are co-optimal
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j+1]):
            # replacement is co-optimal
            if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j]):
                # replacement and copying y[j] are co-optimal
                if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i,j+1]):
                    # replacement, copying y[j], and copying x[i] are co-optimal
                    # Select whether to proceed in any direction uniformly at random
                    r = random.randrange(3)
                    if(r == 0):
                        i += 1
                        j += 1
                    elif(r == 1):
                        i += 1
                    else:
                        j += 1
                else:
                    # select whether to proceed in j direction according to a
                    # coin toss
                    i += 1
                    j += random.randrange(2)
            elif(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i,j+1]):
                # replacement and copying x[i] are co-optimal
                # select whether to proceed in i direction according to a
                # coin toss
                i += random.randrange(2)
                j += 1
            else:
                # only replacement is co-optimal
                i += 1
                j += 1
        elif(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j]):
            # copying y[j] is co-optimal
            if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i,j+1]):
                # copying y[j] and copying x[i] are co-optimal
                # Select whether to proceed in i or j direction uniformly at random
                r = random.randrange(2)
                if(r == 0):
                    i += 1
                else:
                    j += 1
            else:
                # only copying y[j] is co-optimal
                i += 1
        elif(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i,j+1]):
            # only copying x[i] is co-optimal
            j += 1
        else:
            # if we got here, nothing is co-optimal, which is an error
            raise ValueError('Internal error: No option is co-optimal.')
    while(i < m - 1):
        alignment.append_tuple(i, j)
        i += 1
    while(j < n - 1):
        alignment.append_tuple(i, j)
        j += 1
    alignment.append_tuple(m-1, n-1)
    return alignment

