#!python
#cython: language_level=3
"""
Provides a set edit distance to compare two sets (each represented as
lists for computational ease).

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

cimport cython
import numpy as np
from scipy.optimize import linear_sum_assignment
from edist.alignment import Alignment

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.2'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def seted(x, y, delta = None):
    """ Computes the set edit distance between the input set x and the input
    set y, given the element-wise distance function delta.

    In more detail, this function finds an alignment M = {(i, j)} subset of
    {1, ..., len(x)} x {1, ..., len(y)}, such that the following loss is
    minimized:

    .. math:: \\sum_{(i, j) \\in M} \\delta(x_i, y_j) + \\sum_{i \\notin M} \\delta(x_i, -) + \\sum_{j \\notin M} \\delta(-, y_j)

    where i is said to be not in M if there exists no tuple (i, j) in M for
    any j, and j is said to be not in M if there exists no tuple (i, j) in M
    for any i.

    This problem can be solved via the Hungarian or Munkres algorithm in
    O([len(x) + len(y)]^3). So be advised that this method can become very slow
    for large input sets.

    Parameters
    ----------
    x: list-like
        a set of objects.
    y: list-like
        another set of objects.
    delta: function (default = None)
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them. If not given,
        standard_seted is called instead.

    Returns
    -------
    d: float
        the set edit distance between x and y according to delta.

    """
    if(delta is None):
        return standard_seted(x, y)

    d, J = _seted(x, y, delta)
    return d

def _seted(x, y, delta):
    """ Internal function, use seted instead.
    """

    cdef int m = len(x)
    cdef int n = len(y)
    # compute all replacement, deletion, and insertion costs
    Delta = np.zeros((m+1, n+1))
    cdef double[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i, j] = delta(x[i], y[j])
    for i in range(m):
        Delta_view[i, n] = delta(x[i], None)
    for j in range(n):
        Delta_view[m, j] = delta(None, y[j])
    # compute the set edit distance
    return _seted_matrix(Delta)

def _seted_matrix(double[:, :] Delta):
    """ Computes the set edit distance between the input set x and the input
    set y, given the pairwise replacement costs as well as deletion and
    insertion costs.

    In more detail, this function finds an alignment M = {(i, j)} subset of
    {1, ..., m} x {1, ..., n}, where m = len(x) and n = len(y), such that the
    following loss is minimized:

    .. math:: \\sum_{(i, j) \\in M} \\Delta_{i, j} + \\sum_{i \\notin M} \\Delta_{i, n} + \\sum_{j \\notin M} \\Delta_{m, j}

    where i is said to be not in M if there exists no tuple (i, j) in M for
    any j, and j is said to be not in M if there exists no tuple (i, j) in M
    for any i.

    This problem can be solved via the Hungarian or Munkres algorithm in
    O([len(x) + len(y)]^3). So be advised that this method can become very slow
    for large input sets.

    Parameters
    ----------
    Delta: double matrix
        A m + 1 x n + 1 matrix, where Delta[i, j] for i < m and j < n is the
        cost of replacing x[i] with y[j], where Delta[i, n] is the deletion
        cost for x[i], and where Delta[m, j] is the insertion cost for y[j].

    Returns
    -------
    d: float
        the set edit distance between x and y according to Delta.
    J: int array
        J[i] is the index j to which i is mapped. If J[i] is at least n, i
        is deleted. Inversely, if some index j is not contained in J, it is
        inserted.

    """
    # retrieve
    cdef int m = Delta.shape[0] - 1
    cdef int n = Delta.shape[1] - 1
    # set up a cost matrix C for the Hungarian algorithm
    C = np.full((m + n, m + n), np.inf)
    cdef double[:,:] C_view = C
    # set the submatrix 1:m, 1:n to the replacement costs
    C_view[:m, :n] = Delta[:m, :n]
    # set the alignment cost of i with n+i to the deletion costs
    cdef int i
    for i in range(m):
        C_view[i, n+i] = Delta[i, n]
    # set the alignment cost of m+j with j to the insertion costs
    cdef int j
    for j in range(n):
        C_view[m+j, j] = Delta[m, j]
    # set the alignment costs for m+j with n+i to zero
    C_view[m:, n:] = 0.
    # then, apply the Hungarian algorithm
    I, J = linear_sum_assignment(C)
    # and recover the edit distance
    d = np.sum(C[I, J])
    return d, J

def seted_backtrace(x, y, delta = None):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the element-wise distance function delta. This mechanism
    is deterministic and depends on the implementation of
    scipy.optimize.linear_sum_assignment for the choice of co-optimal
    alignment.

    Parameters
    ----------
    x: list-like
        a set of objects.
    y: list-like
        another set of objects.
    delta: function (default = None)
        a function that takes an element of x as first and an element of y
        as second input and returns the distance between them. If None, this
        method calls standard_seted_backtrace instead.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the set edit
        distance.

    """
    if delta is None:
        return standard_seted_backtrace(x, y)

    d, J = _seted(x, y, delta)
    return _to_alignment(J, len(x), len(y))

def _to_alignment(long[:] J, int m, int n):
    """ Transforms a given mapping to an alignment object.

    Parameters
    ----------
    J: int array
        J[i] is the index j to which i is mapped. If J[i] is at least n, i
        is deleted. Inversely, if some index j is not contained in J, it is
        inserted.
    m: int
        the size of the first input set.
    n: int
        the size of the second input set.

    Returns
    -------
    alignment: class alignment.Alignment
        the alignment object corresponding to J.

    """
    alignment = Alignment()
    cdef int i
    for i in range(m):
        if J[i] < n:
            alignment.append_tuple(i, J[i])
        else:
            alignment.append_tuple(i, -1)
    cdef int j
    for j in range(n):
        if J[m+j] < n:
            alignment.append_tuple(-1, j)
    return alignment

def standard_seted(x, y):
    """ Computes the standard set edit distance between the input set x and
    the input set y according to the Kronecker distance, i.e. the replacement
    cost is zero if two elements are equal and one otherwise.

    In more detail, this function finds an alignment M = {(i, j)} subset of
    {1, ..., len(x)} x {1, ..., len(y)}, such that the following loss is
    minimized:

    .. math:: \\sum_{(i, j) \\in M} \\delta(x_i, y_j) + |{i \\notin M}| + |{j \\notin M}|

    where i is said to be not in M if there exists no tuple (i, j) in M for
    any j, and j is said to be not in M if there exists no tuple (i, j) in M
    for any i.

    This problem can be solved via the Hungarian or Munkres algorithm in
    O([len(x) + len(y)]^3). So be advised that this method can become very slow
    for large input sets.

    Parameters
    ----------
    x: list-like
        a set of objects.
    y: list-like
        another set of objects.

    Returns
    -------
    d: float
        the set edit distance between x and y according to delta.

    """
    d, J = _standard_seted(x, y)
    return d

def _standard_seted(x, y):
    """ Internal function, use standard_seted instead.
    """

    cdef int m = len(x)
    cdef int n = len(y)
    # compute all replacement, deletion, and insertion costs
    Delta = np.zeros((m+1, n+1))
    cdef double[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            if x[i] != y[j]:
                Delta_view[i, j] = 1.
    Delta_view[:, n] = 1.
    Delta_view[m, :] = 1.
    return _seted_matrix(Delta)

def standard_seted_backtrace(x, y):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the element-wise distance function delta. This mechanism
    is deterministic and depends on the implementation of
    scipy.optimize.linear_sum_assignment for the choice of co-optimal
    alignment.

    Parameters
    ----------
    x: list-like
        a set of objects.
    y: list-like
        another set of objects.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the standard
        set edit distance.

    """
    d, J = _standard_seted(x, y)
    return _to_alignment(J, len(x), len(y))
