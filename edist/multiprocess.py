"""
Provides general utility functions to compute pairwise edit distances in
parallel.

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

import multiprocessing as mp
import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2023, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

# The size of batches in parallel processing; this size has revealed
# to be favourable in empiric tests, even for longer sequences.
_BATCH_SIZE = 10

def _batch_dist_with_indices(k, l, dist, X, Y, symmetric = False):
    D = np.zeros((len(X), len(Y)))
    if(not symmetric):
        for k2 in range(len(X)):
            for l2 in range(len(Y)):
                D[k2, l2] = dist(X[k2], Y[l2])
    else:
        for k2 in range(len(X)):
            for l2 in range(k2+1, len(Y)):
                D[k2, l2] = dist(X[k2], Y[l2])
    return (k, l, D)

def _batch_dist_with_indices_and_delta(k, l, dist, X, Y, delta, symmetric = False):
    D = np.zeros((len(X), len(Y)))
    if(not symmetric):
        for k2 in range(len(X)):
            for l2 in range(len(Y)):
                D[k2, l2] = dist(X[k2], Y[l2], delta = delta)
    else:
        for k2 in range(len(X)):
            for l2 in range(k2+1, len(Y)):
                D[k2, l2] = dist(X[k2], Y[l2], delta = delta)
    return (k, l, D)

def pairwise_distances(Xs, Ys, dist, delta = None, num_jobs = 8):
    """ Computes the pairwise edit distances between the objects in
    Xs and the objects in Ys. Each object in Xs and Ys needs to be a valid
    input for the given distance function, i.e. a sequence or a tree.

    Optionally, it is possible to specify a component-wise distance function
    delta, which will then be forwarded to the input distance function

    Parameters
    ----------
    Xs: list
        a list of sequences or trees.
    Ys: list
        another list of sequences or trees.
    dist: function
        a function that takes an element of Xs as first and an element of
        Ys as second input and returns a scalar distance value between them.
    delta: function (default = None)
        a function that takes two elements of the input sequences or trees
        as inputs and returns their pairwise distance, where delta(x, None)
        should be the cost of deleting x and delta(None, y) should be the cost
        of inserting y. If this is not None, dist needs to accept an optional
        argument 'delta' as well. Defaults to None.
    num_jobs: int (default = 8)
        The number of jobs to be used for parallel processing. Defaults to 8.

    Returns
    -------
    D: array_like
        a len(Xs) x len(Ys) matrix of pairwise edit distance values.

    """
    K = len(Xs)
    L = len(Ys)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    D = np.zeros((K,L))

    # start off all parallel processing jobs
    results = []
    if(delta is None):
        for k in range(0, K, _BATCH_SIZE):
            k_hi = min(k + _BATCH_SIZE, K)
            for l in range(0, L, _BATCH_SIZE):
                l_hi = min(l + _BATCH_SIZE, L)
                results.append(pool.apply_async(_batch_dist_with_indices, args=(k, l, dist, Xs[k:k_hi], Ys[l:l_hi])))
    else:
        for k in range(0, K, _BATCH_SIZE):
            k_hi = min(k + _BATCH_SIZE, K)
            for l in range(0, L, _BATCH_SIZE):
                l_hi = min(l + _BATCH_SIZE, L)
                results.append(pool.apply_async(_batch_dist_with_indices_and_delta, args=(k, l, dist, Xs[k:k_hi], Ys[l:l_hi], delta)))

    # close the pool
    pool.close()

    # sort all results into the matrix as soon as they arrive
    for res in results:
        k, l, D_batch = res.get()
        k_hi = k + D_batch.shape[0]
        l_hi = l + D_batch.shape[1]
        D[k:k_hi, :][:, l:l_hi] = D_batch

    # ensure that the entire pool is finished
    pool.join()

    # return the distance matrix
    return D

def pairwise_distances_symmetric(Xs, dist, delta = None, num_jobs = 8):
    """ Computes the pairwise edit distances between the objects in
    Xs, assuming that the distance measure is symmetric. Each object in Xs
    needs to be a valid input for the given distance function, i.e. a sequence
    or a tree. Due to symmetry, this method is about double as fast compared
    to pairwise_distances.

    Optionally, it is possible to specify a component-wise distance function
    delta, which will then be forwarded to the input distance function

    Parameters
    ----------
    Xs: list
        a list of sequences or trees.
    dist: function
        a function that takes two elements of Xs as inputs and returns a
        scalar distance value between them.
    delta: function (default = None)
        a function that takes two elements of the input sequences or trees
        as inputs and returns their pairwise distance, where delta(x, None)
        should be the cost of deleting x and delta(None, y) should be the cost
        of inserting y. If this is not None, dist needs to accept an optional
        argument 'delta' as well. Defaults to None.
    num_jobs: int (default = 8)
        The number of jobs to be used for parallel processing. Defaults to 8.

    Returns
    -------
    D: array_like
        a symmetric len(Xs) x len(Xs) matrix of pairwise edit distance values.

    """
    K = len(Xs)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    D = np.zeros((K,K))

    # start off all parallel processing jobs.
    # In each job, we compute a _BATCH_SIZE x _BATCH_SIZE patch of the final
    # pairwise distance matrix. Computing batches reduces the overhead of
    # serialization for starting a parallel processing job, because every
    # single worker can do more with the resources it gets.
    results = []
    if(delta is None):
        for k in range(0, K, _BATCH_SIZE):
            k_hi = min(k + _BATCH_SIZE, K)
            for l in range(k, K, _BATCH_SIZE):
                l_hi = min(l + _BATCH_SIZE, K)
                symmetric = l == k
                results.append(pool.apply_async(_batch_dist_with_indices, args=(k, l, dist, Xs[k:k_hi], Xs[l:l_hi], symmetric)))
    else:
        for k in range(0, K, _BATCH_SIZE):
            k_hi = min(k + _BATCH_SIZE, K)
            for l in range(k, K, _BATCH_SIZE):
                l_hi = min(l + _BATCH_SIZE, K)
                symmetric = l == k
                results.append(pool.apply_async(_batch_dist_with_indices_and_delta, args=(k, l, dist, Xs[k:k_hi], Xs[l:l_hi], delta, symmetric)))

    # close the pool
    pool.close()

    # sort all results into the matrix as soon as they arrive
    for res in results:
        k, l, D_batch = res.get()
        k_hi = k + D_batch.shape[0]
        l_hi = l + D_batch.shape[1]
        D[k:k_hi, :][:, l:l_hi] = D_batch

    # ensure that the entire pool is finished
    pool.join()

    # add the lower diagonal
    D += np.transpose(D)

    # return the distance matrix
    return D

def _batch_backtrace_with_indices(k, l, dist, X, Y):
    B = []
    for k2 in range(len(X)):
        b_k2 = []
        for l2 in range(len(Y)):
            b_k2.append(dist(X[k2], Y[l2]))
        B.append(b_k2)
    return (k, l, B)

def _batch_backtrace_with_indices_and_delta(k, l, dist, X, Y, delta):
    B = []
    for k2 in range(len(X)):
        b_k2 = []
        for l2 in range(len(Y)):
            b_k2.append(dist(X[k2], Y[l2], delta))
        B.append(b_k2)
    return (k, l, B)

def pairwise_backtraces(Xs, Ys, dist_backtrace, delta = None, num_jobs = 8):
    """ Computes the pairwise backtraces between the objects in
    Xs and the objects in Ys. Each object in Xs and Ys needs to be a valid
    input for the given distance function, i.e. a sequence or a tree.

    Optionally, it is possible to specify a component-wise distance function
    delta, which will then be forwarded to the input distance function

    Parameters
    ----------
    Xs: list
        a list of sequences or trees.
    Ys: list
        another list of sequences or trees.
    dist_backtrace: function
        a function that takes an element of Xs as first and an
        element of Ys as second input and returns an arbitrary object.
    delta: function (default = None)
        a function that takes two elements of the input sequences or trees
        as inputs and returns their pairwise distance, where
        delta(x, None) should be the cost of deleting x and delta(None, y)
        should be the cost of inserting y. If this is not None, dist needs
        to accept an optional argument 'delta' as well. Defaults to None.
    num_jobs: int (default = 8)
        The number of jobs to be used for parallel processing. Defaults to 8.

    Returns
    -------
    B: list
        a len(Xs) x len(Ys) list of lists of pairwise backtraces.

    """
    K = len(Xs)
    L = len(Ys)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    B = []
    for k in range(K):
        bs_k = [None] * L
        B.append(bs_k)

    # start off all parallel processing jobs
    results = []
    if(delta is None):
        for k in range(0, K, _BATCH_SIZE):
            k_hi = min(k + _BATCH_SIZE, K)
            for l in range(0, L, _BATCH_SIZE):
                l_hi = min(l + _BATCH_SIZE, L)
                results.append(pool.apply_async(_batch_backtrace_with_indices, args=(k, l, dist_backtrace, Xs[k:k_hi], Ys[l:l_hi])))
    else:
        for k in range(0, K, _BATCH_SIZE):
            k_hi = min(k + _BATCH_SIZE, K)
            for l in range(0, L, _BATCH_SIZE):
                l_hi = min(l + _BATCH_SIZE, L)
                results.append(pool.apply_async(_batch_backtrace_with_indices_and_delta, args=(k, l, dist_backtrace, Xs[k:k_hi], Ys[l:l_hi], delta)))

    # close the pool
    pool.close()

    # sort all results into the matrix as soon as they arrive
    for res in results:
        k, l, B_batch = res.get()
        for k2 in range(len(B_batch)):
            for l2 in range(len(B_batch[k2])):
                B[k + k2][l + l2] = B_batch[k2][l2]

    # ensure that the entire pool is finished
    pool.join()

    # return the distance matrix
    return B
