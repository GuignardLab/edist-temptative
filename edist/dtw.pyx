#!python
#cython: language_level=3
"""
Implements the dynamic time warping distance of Vintsyuk (1968) and its
backtracing in cython.

"""
# Copyright (C) 2019-2021
# Benjamin Paaßen
# AG Machine Learning
# Bielefeld University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import heapq
import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt
cimport cython
from edist.alignment import Alignment

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.2'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def dtw(x, y, delta):
    """ Computes the dynamic time warping distance between the input sequence
    x and the input sequence y, given the element-wise distance function delta.

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them.

    Returns
    -------
    d: float
        the dynamic time warping distance between x and y according to delta.

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

    # Then, compute the dynamic time warping distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_numeric(double[:] x, double[:] y):
    """ Computes the dynamic time warping distance between two input arrays x
    and y, using the absolute value as element-wise distance measure.

    Parameters
    ----------
    x: array_like
        an array of doubles.
    y: array_like
        another array of doubles.

    Returns
    -------
    d: float
        the dynamic time warping distance between x and y.

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
    # Then, compute the dynamic time warping distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_manhattan(double[:,:] x, double[:,:] y):
    """ Computes the multivariate dynamic time warping distance between two
    input arrays x and y, using the Manhattan distance as element-wise
    distance measure.

    Parameters
    ----------
    x: array_like
        a m x K matrix of doubles.
    y: array_like
        a n x K matrix of doubles.

    Returns
    -------
    d: float
        the dynamic time warping distance between x and y.

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
    # Then, compute the dynamic time warping distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_euclidean(double[:,:] x, double[:,:] y):
    """ Computes the multivariate dynamic time warping distance between two
    input arrays x and y, using the Euclidean distance as element-wise
    distance measure.

    Parameters
    ----------
    x: array_like
        a m x K matrix of doubles.
    y: array_like
        a n x K matrix of doubles.

    Returns
    -------
    d: float
        the dynamic time warping distance between x and y.

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
    # Then, compute the dynamic time warping distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
def dtw_string(str x, str y):
    """ Computes the dynamic time warping distance between two
    input strings x and y, using the Kronecker distance as element-wise
    distance measure.

    Parameters
    ----------
    x: str
        a string.
    y: str
        another string.

    Returns
    -------
    d: float
        the dynamic time warping distance between x and y.

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
    for i in prange(m, nogil=True):
        for j in prange(n):
            if(x[i] == y[j]):
                Delta_view[i, j] = 0.
            else:
                Delta_view[i, j] = 1.
    # Then, compute the dynamic time warping distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
cdef void dtw_c(const double[:,:] Delta, double[:,:] D) noexcept nogil:
    """ Computes the dynamic time warping distance between two input sequences
    with pairwise element distances Delta and an (empty) dynamic programming
    matrix D.

    Parameters
    ----------
    Delta: array_like
        a m x n matrix containing the pairwise element distances.
    D: array:like
        another m x n matrix to which the output will be written.
        The dynamic time warping distance will be in cell [0, 0] after the
        computation is finished.

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

    Parameters
    ----------
    a: double
        a number
    b: double
        another number
    c: double
        yet another number

    Returns
    -------
    min3: double
        min({a, b, c})

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

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to dynamic time
        warping.

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

    # Then, compute the dynamic time warping distance
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

def dtw_backtrace_stochastic(x, y, delta):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the element-wise distance function delta. This mechanism
    is stochastic and will return a random co-optimal alignment.

    Note that the randomness does _not_ produce a uniform distribution over
    all co-optimal alignments because reandom choices at the start of the
    alignment process dominate. If you wish to characterize the overall
    distribution accurately, use sed_backtrace_matrix instead.

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to dynamic time
        warping.

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

    # Then, compute the dynamic time warping distance
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

def dtw_backtrace_matrix(x, y, delta):
    """ Computes a matrix, summarizing all co-optimal alignments between
    x and y in a matrix P, where entry P[i, j] specifies the fraction of
    co-optimal alignments in which node x[i] has been aligned with node y[j].

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them.

    Returns
    -------
    P: array_like
        a matrix, where entry P[i, j] specifies the fraction of co-optimal
        alignments in which node x[i] has been aligned with node y[j].
    K: array_like
        a matrix that contains the counts for all co-optimal alignments in
        which node x[i] has been aligned with node y[j].
    k: int
        the number of co-optimal alignments overall, such that P = K / k.

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

    # Then, compute the dynamic time warping distance
    D = np.zeros((m,n))
    dtw_c(Delta, D)
    cdef double[:,:] D_view = D

    # compute the forward matrix Alpha, which contains the number of
    # co-optimal alignment paths from cell [0, 0] to cell [i, j]
    Alpha = np.zeros((m, n), dtype=int)
    cdef long[:,:] Alpha_view = Alpha
    Alpha_view[0, 0] = 1
    # build a queue of cells which we still need to process
    q = [(0, 0)]
    # build a set which stores the already visited cells
    visited = set()
    # initialize temporary variables
    cdef int found_coopt = False
    cdef long k = 0
    while(q):
        (i, j) = heapq.heappop(q)
        if((i, j) in visited):
            continue
        visited.add((i, j))
        k = Alpha_view[i, j]
        if(i == m-1):
            if(j == n-1):
                continue
            # if we are at the end of the first sequence, we can only copy
            # that end
            Alpha_view[i, j+1] += k
            heapq.heappush(q, (i, j+1))
            continue
        if(j == n-1):
            # if we are at the end of the second sequence, we can only copy
            # that end
            Alpha_view[i+1, j] += k
            heapq.heappush(q, (i+1, j))
            continue
        found_coopt = False
        # check which alignment option is co-optimal
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j+1]):
            # replacement is co-optimal
            Alpha_view[i+1, j+1] += k
            heapq.heappush(q, (i+1, j+1))
            found_coopt = True
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j]):
            # copying y[j] is co-optimal
            Alpha_view[i+1, j] += k
            heapq.heappush(q, (i+1, j))
            found_coopt = True
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i,j+1]):
            # copying x[i] is co-optimal
            Alpha_view[i, j+1] += k
            heapq.heappush(q, (i, j+1))
            found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    # compute the backward matrix Beta, which contains the number of
    # co-optimal alignment paths from cell [i, j] to cell [m-1, n-1]
    Beta = np.zeros((m, n), dtype=int)
    cdef long[:,:] Beta_view = Beta
    Beta_view[m-1, n-1] = 1
    # iterate in downward lexigraphic order over the visited cells
    for (i, j) in sorted(visited, reverse = True):
        k = Beta_view[i, j]
        if(i == 0):
            if(j == 0):
                continue
            # if we are at the start of the first sequence, we can only copy
            # this start
            Beta_view[i, j-1] += k
            continue
        if(j == 0):
            # if we are at the start of the second sequence, we can only copy
            # this start
            Beta_view[i-1, j] += k
            continue
        found_coopt = False
        # check which alignment option is co-optimal
        if(D_view[i-1,j-1] + _BACKTRACE_TOL > Delta_view[i-1,j-1] + D_view[i,j]):
            # replacement is co-optimal
            Beta_view[i-1, j-1] += k
            found_coopt = True
        if(D_view[i-1,j] + _BACKTRACE_TOL > Delta_view[i-1,j] + D_view[i,j]):
            # copying y[j] is co-optimal
            Beta_view[i-1, j] += k
            found_coopt = True
        if(D_view[i,j-1] + _BACKTRACE_TOL > Delta_view[i,j-1] + D_view[i,j]):
            # copying x[i] is co-optimal
            Beta_view[i, j-1] += k
            found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    if(Alpha_view[m-1, n-1] != Beta_view[0, 0]):
        raise ValueError('Internal error: Alignment count in Alpha and Beta matrix did not agree; got %d versus %d' % (Alpha_view[m-1, n-1], Beta_view[0, 0]))

    # compute a counting matrix specifying how often each alignment has
    # occured by multiplying alpha and beta values.
    K = Alpha * Beta
    # compute the final summary matrix by dividing K by the overall number
    # of co-optimal alignments
    return K.astype(np.float) / Alpha_view[m-1, n-1], K, Alpha_view[m-1, n-1]

