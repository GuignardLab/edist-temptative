"""
Provides general utility functions to compute pairwise edit distances in
parallel.

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

import multiprocessing as mp
import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def dist_with_indices(k, l, dist, x, y):
    return (k, l, dist(x, y))

def dist_with_indices_and_delta(k, l, dist, x, y, delta):
    return (k, l, dist(x, y, delta = delta))

def pairwise_distances(Xs, Ys, dist, delta = None, num_jobs = 8):
    """ Computes the pairwise edit distances between the objects in
    Xs and the objects in Ys. Each object in Xs and Ys needs to be a valid
    input for the given distance function, i.e. a sequence or a tree.

    Optionally, it is possible to specify a component-wise distance function
    delta, which will then be forwarded to the input distance function

    Args:
    Xs:      a list of sequences or trees.
    Ys:      another list of sequences or trees.
    dist:    a function that takes an element of Xs as first and an element of
             Ys as second input and returns a scalar distance value between
             them.
    delta:   a function that takes two elements of the input sequences or trees
             as inputs and returns their pairwise distance, where
             delta(x, None) should be the cost of deleting x and delta(None, y)
             should be the cost of inserting y. If this is not None, dist needs
             to accept an optional argument 'delta' as well. Defaults to None.
    num_jobs: The number of jobs to be used for parallel processing.
             Defaults to 8.

    Returns: a len(Xs) x len(Ys) matrix of pairwise edit distance values.
    """
    K = len(Xs)
    L = len(Ys)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    D = np.zeros((K,L))

    # set up the callback function
    def callback(tpl):
        D[tpl[0], tpl[1]] = tpl[2]

    # start off all parallel processing jobs
    if(not delta):
        for k in range(K):
            for l in range(L):
                pool.apply_async(dist_with_indices, args=(k, l, dist, Xs[k], Ys[l]), callback=callback)
    else:
        for k in range(K):
            for l in range(L):
                pool.apply_async(dist_with_indices_and_delta, args=(k, l, dist, Xs[k], Ys[l], delta), callback=callback)

    # wait for the jobs to finish
    pool.close()
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

    Args:
    Xs:      a list of sequences or trees.
    dist:    a function that takes two elements of Xs as first and second input
             and returns a scalar distance value between them.
    delta:   a function that takes two elements of the input sequences or trees
             as inputs and returns their pairwise distance, where
             delta(x, None) should be the cost of deleting x and delta(None, y)
             should be the cost of inserting y. If this is not None, dist needs
             to accept an optional argument 'delta' as well. Defaults to None.
    num_jobs: The number of jobs to be used for parallel processing.
             Defaults to 8.

    Returns: a symmetric len(Xs) x len(Xs) matrix of pairwise edit distance
             values.
    """
    K = len(Xs)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    D = np.zeros((K,K))

    # set up the callback function
    def callback(tpl):
        D[tpl[0], tpl[1]] = tpl[2]

    # start off all parallel processing jobs
    if(not delta):
        for k in range(K):
            for l in range(k+1, K):
                pool.apply_async(dist_with_indices, args=(k, l, dist, Xs[k], Xs[l]), callback=callback)
    else:
        for k in range(K):
            for l in range(k+1, K):
                pool.apply_async(dist_with_indices_and_delta, args=(k, l, dist, Xs[k], Xs[l], delta), callback=callback)

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # add the lower diagonal
    D += np.transpose(D)

    # return the distance matrix
    return D
