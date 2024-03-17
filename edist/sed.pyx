#!python
#cython: language_level=3
"""
Implements the sequence edit distance of Levenshtein (1965) and its
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


###################################
# Edit Distance with Custom Delta #
###################################

def sed(x, y, delta = None):
    """ Computes the sequence edit distance between the input sequence
    x and the input sequence y, given the element-wise distance function delta.

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function (default = None)
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them. If None, this
        method calls standard_sed instead.

    Returns
    -------
    d: float
        the sequence edit distance between x and y according to delta.

    """
    if(delta is None):
        return float(standard_sed(x, y))
    _, _, _, D = _sed(x, y, delta)
    return D[0,0]

def _sed(x, y, delta):
    """ Internal function. Call sed instead. """
    cdef int m = len(x)
    cdef int n = len(y)
    # First, compute all pairwise replacements
    Delta = np.zeros((m, n))
    cdef double[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i,j] = delta(x[i], y[j])

    # Then, compute all deletions
    Delta_del = np.zeros(m)
    cdef double[:] Delta_del_view = Delta_del
    for i in range(m):
        Delta_del_view[i] = delta(x[i], None)

    # Then, compute all insertions
    Delta_ins = np.zeros(n)
    cdef double[:] Delta_ins_view = Delta_ins
    for j in range(n):
        Delta_ins_view[j] = delta(None, y[j])

    # Then, compute the sequence edit distance
    D = np.zeros((m+1,n+1))
    sed_c(Delta, Delta_del, Delta_ins, D)

    return Delta, Delta_del, Delta_ins, D


@cython.boundscheck(False)
cdef void sed_c(const double[:,:] Delta, const double[:] Delta_del, const double[:] Delta_ins, double[:,:] D) noexcept nogil:
    """ Computes the sequence edit distance between two input sequences
    with pairwise element distances Delta and an (empty) dynamic programming
    matrix D.

    Arguments
    ---------
    Delta: double matrix
        a m x n matrix containing the pairwise element distances.
    Delta_del: double array
        a m-element vector containing deletion costs.
    Delta_ins: double array
        a n-element vector containing insertion costs.
    D: double matrix
        an m+1 x n+1 matrix to which the output will be written.
        The sequence edit distance will be in cell [0, 0] after the computation
        is finished.

    """
    cdef int m = Delta.shape[0]
    cdef int n = Delta.shape[1]
    cdef int i
    cdef int j
    # initialize last entry
    D[m, n] = 0.
    # compute last column
    for i in range(m-1,-1,-1):
        D[i,n] = Delta_del[i] + D[i+1,n]
    # compute last row
    for j in range(n-1,-1,-1):
        D[m,j] = Delta_ins[j] + D[m,j+1]
    # compute remaining matrix
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            D[i,j] = min3(Delta[i,j] + D[i+1,j+1],
                          Delta_del[i] + D[i+1, j],
                          Delta_ins[j] + D[i, j+1])

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

#########################
# Backtracing Functions #
#########################

cdef double _BACKTRACE_TOL = 1E-5

def sed_backtrace(x, y, delta = None):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the element-wise distance function delta. This mechanism
    is deterministic and will always prefer replacements over other options.

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function (default = None)
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them. If None, this
        method calls standard_sed_backtrace instead.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the sequence edit
        distance.

    """
    if(delta is None):
        return standard_sed_backtrace(x, y)
    cdef int m = len(x)
    cdef int n = len(y)
    Delta, Delta_del, Delta_ins, D = _sed(x, y, delta)
    cdef double[:,:] Delta_view = Delta
    cdef double[:] Delta_del_view = Delta_del
    cdef double[:] Delta_ins_view = Delta_ins
    cdef double[:,:] D_view = D

    # Finally, compute the backtrace
    cdef int i = 0
    cdef int j = 0
    alignment = Alignment()
    while(i < m and j < n):
        # check which alignment option is co-optimal
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j+1]):
            # replacement is co-optimal
            alignment.append_tuple(i, j)
            i += 1
            j += 1
            continue
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_del_view[i] + D_view[i+1,j]):
            # deleting x[i] is co-optimal
            alignment.append_tuple(i, -1)
            i += 1
            continue
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_ins_view[j] + D_view[i,j+1]):
            # inserting y[j] is co-optimal
            alignment.append_tuple(-1, j)
            j += 1
            continue
        # if we got here, nothing is co-optimal, which is an error
        raise ValueError('Internal error: No option is co-optimal.')
    while(i < m):
        alignment.append_tuple(i, -1)
        i += 1
    while(j < n):
        alignment.append_tuple(-1, j)
        j += 1
    return alignment

def sed_backtrace_stochastic(x, y, delta = None):
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
    delta: function (default = None)
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them. If None, this
        method calls standard_sed_backtrace_stochastic instead.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the sequence edit
        distance.

    """
    if(delta is None):
        return standard_sed_backtrace_stochastic(x, y)
    cdef int m = len(x)
    cdef int n = len(y)
    Delta, Delta_del, Delta_ins, D = _sed(x, y, delta)
    cdef double[:,:] Delta_view = Delta
    cdef double[:] Delta_del_view = Delta_del
    cdef double[:] Delta_ins_view = Delta_ins
    cdef double[:,:] D_view = D

    # Finally, compute the backtrace
    cdef int i = 0
    cdef int j = 0
    alignment = Alignment()
    while(i < m and j < n):
        # check which alignment option is co-optimal
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j+1]):
            if(D_view[i,j] + _BACKTRACE_TOL > Delta_del_view[i] + D_view[i+1,j]):
                if(D_view[i,j] + _BACKTRACE_TOL > Delta_ins_view[j] + D_view[i,j+1]):
                    # replacement, deletion, and insertion is co-optimal
                    # Select how to proceed uniformly at random
                    r = random.randrange(3)
                    if(r == 0):
                        alignment.append_tuple(i, j)
                        i += 1
                        j += 1
                    elif(r == 1):
                        alignment.append_tuple(i, -1)
                        i += 1
                    else:
                        alignment.append_tuple(-1, j)
                        j += 1
                else:
                    # replacement and deletion is co-optimal.
                    # Select how to proceed uniformly at random
                    if(random.randrange(2) == 0):
                        alignment.append_tuple(i, j)
                        i += 1
                        j += 1
                    else:
                        alignment.append_tuple(i, -1)
                        i += 1
            elif(D_view[i,j] + _BACKTRACE_TOL > Delta_ins_view[j] + D_view[i,j+1]):
                # replacement and insertion is co-optimal.
                # Select how to proceed uniformly at random
                if(random.randrange(2) == 0):
                    alignment.append_tuple(i, j)
                    i += 1
                    j += 1
                else:
                    alignment.append_tuple(-1, j)
                    j += 1
            else:
                # only replacement is co-optimal
                alignment.append_tuple(i, j)
                i += 1
                j += 1
        elif(D_view[i,j] + _BACKTRACE_TOL > Delta_del_view[i] + D_view[i+1,j]):
            if(D_view[i,j] + _BACKTRACE_TOL > Delta_ins_view[j] + D_view[i,j+1]):
                # deletion and insertion is cooptimal
                # Select how to proceed uniformly at random
                if(random.randrange(2) == 0):
                    alignment.append_tuple(i, -1)
                    i += 1
                else:
                    alignment.append_tuple(-1, j)
                    j += 1
            else:
                # only deletion is co-optimal
                alignment.append_tuple(i, -1)
                i += 1
        elif(D_view[i,j] + _BACKTRACE_TOL > Delta_ins_view[j] + D_view[i,j+1]):
            # only insertion is co-optimal
            alignment.append_tuple(-1, j)
            j += 1
        else:
            # if we got here, nothing is co-optimal, which is an error
            raise ValueError('Internal error: No option is co-optimal.')
    while(i < m):
        alignment.append_tuple(i, -1)
        i += 1
    while(j < n):
        alignment.append_tuple(-1, j)
        j += 1
    return alignment

def sed_backtrace_matrix(x, y, delta = None):
    """ Computes a matrix, summarizing all co-optimal alignments between
    x and y in a matrix P, where entry P[i, j] specifies the fraction of
    co-optimal alignments in which node x[i] has been aligned with node y[j].

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.
    delta: function (default = None)
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them. If None, this
        method calls standard_sed_backtrace_matrix instead.

    Returns
    -------
    P: array_like
        a matrix, where entry P[i, j] specifies the fraction of co-optimal
        alignments in which node x[i] has been aligned with node y[j].
        P[i, n] contains the fraction of deletions of node x[i] and P[m, j]
        the fraction of insertions of node y[j].
    K: array_like
        a matrix that contains the counts for all co-optimal alignments in
        which node x[i] has been aligned with node y[j].
    k: int
        the number of co-optimal alignments overall, such that P = K / k.

    """
    if(delta is None):
        return standard_sed_backtrace_matrix(x, y)
    cdef int m = len(x)
    cdef int n = len(y)
    Delta, Delta_del, Delta_ins, D = _sed(x, y, delta)
    cdef double[:,:] Delta_view = Delta
    cdef double[:] Delta_del_view = Delta_del
    cdef double[:] Delta_ins_view = Delta_ins
    cdef double[:,:] D_view = D

    # compute the forward matrix Alpha, which contains the number of
    # co-optimal alignment paths from cell [0, 0] to cell [i, j]
    Alpha = np.zeros((m+1, n+1), dtype=int)
    cdef long[:,:] Alpha_view = Alpha
    Alpha_view[0, 0] = 1
    # build a queue of cells which we still need to process
    q = [(0, 0)]
    # build a set which stores the already visited cells
    visited = set()
    # initialize temporary variables
    cdef int found_coopt = False
    cdef int i
    cdef int j
    cdef long k = 0
    while(q):
        (i, j) = heapq.heappop(q)
        if((i, j) in visited):
            continue
        visited.add((i, j))
        k = Alpha_view[i, j]
        if(i == m):
            if(j == n):
                continue
            # if we are at the end of the first sequence, we can only insert
            Alpha_view[i, j+1] += k
            heapq.heappush(q, (i, j+1))
            continue
        if(j == n):
            # if we are at the end of the second sequence, we can only delete
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
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_del_view[i] + D_view[i+1,j]):
            # deletion is co-optimal
            Alpha_view[i+1, j] += k
            heapq.heappush(q, (i+1, j))
            found_coopt = True
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_ins_view[j] + D_view[i,j+1]):
            # insertion is co-optimal
            Alpha_view[i, j+1] += k
            heapq.heappush(q, (i, j+1))
            found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    # compute the backward matrix Beta, which contains the number of
    # co-optimal alignment paths from cell [i, j] to cell [m, n]
    Beta = np.zeros((m+1, n+1), dtype=int)
    cdef long[:,:] Beta_view = Beta
    Beta_view[m, n] = 1
    # iterate in downward lexigraphic order over the visited cells
    for (i, j) in sorted(visited, reverse = True):
        k = Beta_view[i, j]
        if(i == 0):
            if(j == 0):
                continue
            # if we are at the start of the first sequence, we can only insert
            Beta_view[i, j-1] += k
            continue
        if(j == 0):
            # if we are at the start of the second sequence, we can only delete
            Beta_view[i-1, j] += k
            continue
        found_coopt = False
        # check which alignment option is co-optimal
        if(D_view[i-1,j-1] + _BACKTRACE_TOL > Delta_view[i-1,j-1] + D_view[i,j]):
            # replacement is co-optimal
            Beta_view[i-1, j-1] += k
            found_coopt = True
        if(D_view[i-1,j] + _BACKTRACE_TOL > Delta_del_view[i-1] + D_view[i,j]):
            # deletion is co-optimal
            Beta_view[i-1, j] += k
            found_coopt = True
        if(D_view[i,j-1] + _BACKTRACE_TOL > Delta_ins_view[j-1] + D_view[i,j]):
            # insertion is co-optimal
            Beta_view[i, j-1] += k
            found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    if(Alpha_view[m, n] != Beta_view[0, 0]):
        raise ValueError('Internal error: Alignment count in Alpha and Beta matrix did not agree; got %d versus %d' % (Alpha_view[m, n], Beta_view[0, 0]))

    # compute a counting matrix specifying how often each alignment has
    # occured by multiplying alpha and beta values.
    K = np.zeros((m, n), dtype=int)
    cdef long[:,:] K_view = K
    for (i, j) in visited:
        if(i == m or j == n):
            continue
        # check if replacement is co-optimal
        if(D_view[i,j] + _BACKTRACE_TOL > Delta_view[i,j] + D_view[i+1,j+1]):
            K_view[i, j] = Alpha_view[i, j] * Beta_view[i+1, j+1]

    # compute the final summary matrix by dividing K by the overall number
    # of co-optimal alignments and completing the last row and column
    k = Alpha_view[m, n]
    P = np.zeros((m+1, n+1))
    P[:m, :][:, :n] = K
    P[:m, n] = k - np.sum(K, axis=1)
    P[m, :n] = k - np.sum(K, axis=0)
    P /= k

    return P, K, k

###############################################
# Standard Edit Distance with Kronecker Delta #
###############################################

def standard_sed(x, y):
    """ Computes the standard sequence edit distance/Levenshtein distance
    between the input sequence x and the input sequence y.

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.

    Returns
    -------
    d: int
        the standard sequence edit distance between x and y.

    """
    _, D = _standard_sed(x, y)
    return D[0, 0]

def _standard_sed(x, y):
    """ Internal function. Call standard_sed instead. """
    cdef int m = len(x)
    cdef int n = len(y)
    # First, compute all pairwise replacements
    Delta = np.zeros((m, n), dtype=int)
    cdef long[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            if(x[i] != y[j]):
                Delta_view[i, j] = 1

    # Then, compute the sequence edit distance
    D = np.zeros((m+1,n+1), dtype=int)
    standard_sed_c(Delta, D)
    return Delta, D

@cython.boundscheck(False)
def sed_string(str x, str y):
    """ Computes the standard sequence edit distance/Levenshtein distance
    between two input strings x and y, using the Kronecker distance as
    element-wise distance measure.

    Parameters
    ----------
    x: str
        a string.
    y: str
        another string.

    Returns
    -------
    d: int
        the standard sequence edit distance between x and y.

    """
    cdef int m = len(x)
    cdef int n = len(y)
    # First, compute all pairwise replacements
    Delta = np.zeros((m, n), dtype=int)
    cdef long[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in prange(m, nogil=True):
        for j in prange(n):
            if(x[i] != y[j]):
                Delta_view[i, j] = 1

    # Then, compute the standard sequence edit distance
    D = np.zeros((m+1,n+1), dtype=int)
    standard_sed_c(Delta, D)
    return D[0,0]

@cython.boundscheck(False)
cdef void standard_sed_c(const long[:,:] Delta, long[:,:] D) noexcept nogil:
    """ Computes the standard sequence edit distance between two input sequences
    with pairwise element distances Delta and an (empty) dynamic programming
    matrix D.

    Parameters
    ----------
    Delta: long matrix
        a m x n matrix containing the pairwise element distances.
    D: long matrix
        another m x n matrix to which the output will be written.
        The sequence edit distance will be in cell [0, 0] after the computation
        is finished.

    """
    cdef int m = Delta.shape[0]
    cdef int n = Delta.shape[1]
    cdef int i
    cdef int j
    # initialize last entry
    D[m, n] = 0
    # compute last column
    for i in range(m-1,-1,-1):
        D[i,n] = 1 + D[i+1,n]
    # compute last row
    for j in range(n-1,-1,-1):
        D[m,j] = 1 + D[m,j+1]
    # compute remaining matrix
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            D[i,j] = min3_int(Delta[i,j] + D[i+1,j+1],
                          1 + D[i+1, j],
                          1 + D[i, j+1])

cdef long min3_int(long a, long b, long c) nogil:
    """ Computes the minimum of three numbers.

    Parameters
    ----------
    a: int
        a number
    b: int
        another number
    c: int
        yet another number

    Returns
    -------
    min3: int
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

#########################
# Backtracing Functions #
#########################

def standard_sed_backtrace(x, y):
    """ Computes a co-optimal alignment between the two input sequences
    x and y. This mechanism is deterministic and will always prefer
    replacements over other options.

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the sequence edit
        distance.

    """
    cdef int m = len(x)
    cdef int n = len(y)
    # Compute the standard edit distance first
    Delta, D = _standard_sed(x, y)
    cdef long[:,:] Delta_view = Delta
    cdef long[:,:] D_view = D

    # Then, compute the backtrace
    cdef int i = 0
    cdef int j = 0
    alignment = Alignment()
    while(i < m and j < n):
        # check which alignment option is co-optimal
        if(D_view[i,j] == Delta_view[i,j] + D_view[i+1,j+1]):
            # replacement is co-optimal
            alignment.append_tuple(i, j)
            i += 1
            j += 1
            continue
        if(D_view[i,j] == 1 + D_view[i+1,j]):
            # deleting x[i] is co-optimal
            alignment.append_tuple(i, -1)
            i += 1
            continue
        if(D_view[i,j] == 1 + D_view[i,j+1]):
            # inserting y[j] is co-optimal
            alignment.append_tuple(-1, j)
            j += 1
            continue
        # if we got here, nothing is co-optimal, which is an error
        raise ValueError('Internal error: No option is co-optimal.')
    while(i < m):
        alignment.append_tuple(i, -1)
        i += 1
    while(j < n):
        alignment.append_tuple(-1, j)
        j += 1
    return alignment

def standard_sed_backtrace_stochastic(x, y):
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

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the sequence edit
        distance.

    """

    cdef int m = len(x)
    cdef int n = len(y)
    # Compute the standard edit distance first
    Delta, D = _standard_sed(x, y)
    cdef long[:,:] Delta_view = Delta
    cdef long[:,:] D_view = D

    # Then, compute the backtrace
    cdef int i = 0
    cdef int j = 0
    alignment = Alignment()
    while(i < m and j < n):
        # check which alignment option is co-optimal
        if(D_view[i,j] == Delta_view[i,j] + D_view[i+1,j+1]):
            if(D_view[i,j] == 1 + D_view[i+1,j]):
                if(D_view[i,j] == 1 + D_view[i,j+1]):
                    # replacement, deletion, and insertion is co-optimal
                    # Select how to proceed uniformly at random
                    r = random.randrange(3)
                    if(r == 0):
                        alignment.append_tuple(i, j)
                        i += 1
                        j += 1
                    elif(r == 1):
                        alignment.append_tuple(i, -1)
                        i += 1
                    else:
                        alignment.append_tuple(-1, j)
                        j += 1
                else:
                    # replacement and deletion is co-optimal.
                    # Select how to proceed uniformly at random
                    if(random.randrange(2) == 0):
                        alignment.append_tuple(i, j)
                        i += 1
                        j += 1
                    else:
                        alignment.append_tuple(i, -1)
                        i += 1
            elif(D_view[i,j] == 1 + D_view[i,j+1]):
                # replacement and insertion is co-optimal.
                # Select how to proceed uniformly at random
                if(random.randrange(2) == 0):
                    alignment.append_tuple(i, j)
                    i += 1
                    j += 1
                else:
                    alignment.append_tuple(-1, j)
                    j += 1
            else:
                # only replacement is co-optimal
                alignment.append_tuple(i, j)
                i += 1
                j += 1
        elif(D_view[i,j] == 1 + D_view[i+1,j]):
            if(D_view[i,j] == 1 + D_view[i,j+1]):
                # deletion and insertion is cooptimal
                # Select how to proceed uniformly at random
                if(random.randrange(2) == 0):
                    alignment.append_tuple(i, -1)
                    i += 1
                else:
                    alignment.append_tuple(-1, j)
                    j += 1
            else:
                # only deletion is co-optimal
                alignment.append_tuple(i, -1)
                i += 1
        elif(D_view[i,j] == 1 + D_view[i,j+1]):
            # only insertion is co-optimal
            alignment.append_tuple(-1, j)
            j += 1
        else:
            # if we got here, nothing is co-optimal, which is an error
            raise ValueError('Internal error: No option is co-optimal.')
    while(i < m):
        alignment.append_tuple(i, -1)
        i += 1
    while(j < n):
        alignment.append_tuple(-1, j)
        j += 1
    return alignment

def standard_sed_backtrace_matrix(x, y):
    """ Computes a matrix, summarizing all co-optimal alignments between
    x and y in a matrix P, where entry P[i, j] specifies the fraction of
    co-optimal alignments in which node x[i] has been aligned with node y[j].

    Parameters
    ----------
    x: list
        a sequence of objects.
    y: list
        another sequence of objects.

    Returns
    -------
    P: array_like
        a matrix, where entry P[i, j] specifies the fraction of co-optimal
        alignments in which node x[i] has been aligned with node y[j].
        P[i, n] contains the fraction of deletions of node x[i] and P[m, j]
        the fraction of insertions of node y[j].
    K: array_like
        a matrix that contains the counts for all co-optimal alignments in
        which node x[i] has been aligned with node y[j].
    k: int
        the number of co-optimal alignments overall, such that P = K / k.

    """

    cdef int m = len(x)
    cdef int n = len(y)
    # Compute the standard edit distance first
    Delta, D = _standard_sed(x, y)
    cdef long[:,:] Delta_view = Delta
    cdef long[:,:] D_view = D

    # compute the forward matrix Alpha, which contains the number of
    # co-optimal alignment paths from cell [0, 0] to cell [i, j]
    Alpha = np.zeros((m+1, n+1), dtype=int)
    cdef long[:,:] Alpha_view = Alpha
    Alpha_view[0, 0] = 1
    # build a queue of cells which we still need to process
    q = [(0, 0)]
    # build a set which stores the already visited cells
    visited = set()
    # initialize temporary variables
    cdef int found_coopt = False
    cdef int i
    cdef int j
    cdef long k = 0
    while(q):
        (i, j) = heapq.heappop(q)
        if((i, j) in visited):
            continue
        visited.add((i, j))
        k = Alpha_view[i, j]
        if(i == m):
            if(j == n):
                continue
            # if we are at the end of the first sequence, we can only insert
            Alpha_view[i, j+1] += k
            heapq.heappush(q, (i, j+1))
            continue
        if(j == n):
            # if we are at the end of the second sequence, we can only delete
            Alpha_view[i+1, j] += k
            heapq.heappush(q, (i+1, j))
            continue
        found_coopt = False
        # check which alignment option is co-optimal
        if(D_view[i,j] == Delta_view[i,j] + D_view[i+1,j+1]):
            # replacement is co-optimal
            Alpha_view[i+1, j+1] += k
            heapq.heappush(q, (i+1, j+1))
            found_coopt = True
        if(D_view[i,j] == 1 + D_view[i+1,j]):
            # deletion is co-optimal
            Alpha_view[i+1, j] += k
            heapq.heappush(q, (i+1, j))
            found_coopt = True
        if(D_view[i,j] == 1 + D_view[i,j+1]):
            # insertion is co-optimal
            Alpha_view[i, j+1] += k
            heapq.heappush(q, (i, j+1))
            found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    # compute the backward matrix Beta, which contains the number of
    # co-optimal alignment paths from cell [i, j] to cell [m, n]
    Beta = np.zeros((m+1, n+1), dtype=int)
    cdef long[:,:] Beta_view = Beta
    Beta_view[m, n] = 1
    # iterate in downward lexigraphic order over the visited cells
    for (i, j) in sorted(visited, reverse = True):
        k = Beta_view[i, j]
        if(i == 0):
            if(j == 0):
                continue
            # if we are at the start of the first sequence, we can only insert
            Beta_view[i, j-1] += k
            continue
        if(j == 0):
            # if we are at the start of the second sequence, we can only delete
            Beta_view[i-1, j] += k
            continue
        found_coopt = False
        # check which alignment option is co-optimal
        if(D_view[i-1,j-1] == Delta_view[i-1,j-1] + D_view[i,j]):
            # replacement is co-optimal
            Beta_view[i-1, j-1] += k
            found_coopt = True
        if(D_view[i-1,j] == 1 + D_view[i,j]):
            # deletion is co-optimal
            Beta_view[i-1, j] += k
            found_coopt = True
        if(D_view[i,j-1] == 1 + D_view[i,j]):
            # insertion is co-optimal
            Beta_view[i, j-1] += k
            found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    if(Alpha_view[m, n] != Beta_view[0, 0]):
        raise ValueError('Internal error: Alignment count in Alpha and Beta matrix did not agree; got %d versus %d' % (Alpha_view[m, n], Beta_view[0, 0]))

    # compute a counting matrix specifying how often each alignment has
    # occured by multiplying alpha and beta values.
    K = np.zeros((m, n), dtype=int)
    cdef long[:,:] K_view = K
    for (i, j) in visited:
        if(i == m or j == n):
            continue
        # check if replacement is co-optimal
        if(D_view[i,j] == Delta_view[i,j] + D_view[i+1,j+1]):
            K_view[i, j] = Alpha_view[i, j] * Beta_view[i+1, j+1]

    # compute the final summary matrix by dividing K by the overall number
    # of co-optimal alignments and completing the last row and column
    k = Alpha_view[m, n]
    P = np.zeros((m+1, n+1))
    P[:m, :][:, :n] = K
    P[:m, n] = k - np.sum(K, axis=1)
    P[m, :n] = k - np.sum(K, axis=0)
    P /= k

    return P, K, k

