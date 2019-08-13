#!python
#cython: language_level=3
"""
Implements the tree edit distance and its backtracing in cython.

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
from cpython cimport bool
cimport cython
from edist.alignment import Alignment

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

# The tree edit distance with custom delta

def ted(x_nodes, x_adj, y_nodes = None, y_adj = None, delta = None):
    """ Computes the tree edit distance between the trees x and y, each
    described by a list of nodes and an adjacency list adj, where adj[i]
    is a list of indices pointing to children of node i.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    x_nodes: a list of nodes for tree x
             OR a tuple of the form (x_nodes, x_adj).
    x_adj:   an adjacency list for tree x
             OR a tuple of the form (y_nodes, y_adj).
    y_nodes: a list of nodes for tree y.
    y_adj:   an adjacency list for tree y.
    delta:   a function that takes two nodes as inputs and returns their
             pairwise distance, where delta(x, None) should be the cost of
             deleting x and delta(None, y) should be the cost of inserting y.
             If undefined, this method calls standard_ted instead.

    Returns: the tree edit distance between x and y according to delta.
    """
    if(delta is None):
        return float(standard_ted(x_nodes, x_adj, y_nodes, y_adj))

    if(isinstance(x_nodes, tuple)):
        x_nodes, x_adj, y_nodes, y_adj = extract_from_tuple_input(x_nodes, x_adj)

    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)
    # if both trees are empty, the distance is necessarily zero.
    if(m == 0 and n == 0):
        return 0.
    # An array to store all edit costs for replacements, deletions, and
    # insertions
    Delta = np.zeros((m+1, n+1))
    cdef double[:,:] Delta_view = Delta
    # First, compute all pairwise replacement costs
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i,j] = delta(x_nodes[i], y_nodes[j])

    # Then, compute the deletion and insertion costs
    for i in range(m):
        Delta_view[i,n] = delta(x_nodes[i], None)
    for j in range(n):
        Delta_view[m,j] = delta(None, y_nodes[j])

    # if either tree is empty, we can only delete/insert all nodes in the
    # non-empty tree.
    if(m == 0):
        return np.sum(Delta[0,:])
    if(n == 0):
        return np.sum(Delta[:,0])

    # Compute the keyroots and outermost right leaves for both trees.
    x_orl = outermost_right_leaves(x_adj)
    x_kr  = keyroots(x_orl)
    y_orl = outermost_right_leaves(y_adj)
    y_kr  = keyroots(y_orl)

    # Finally, compute the actual tree edit distance
    D_forest = np.zeros((m+1,n+1))
    D_tree = np.zeros((m,n))
    _ted_c(x_orl, x_kr, y_orl, y_kr, Delta, D_forest, D_tree)
    return D_tree[0,0]

def extract_from_tuple_input(x, y):
    if(len(x) != 2):
        raise ValueError('If the first input is a tuple, it needs to contain exactly two elements, a node list and an adjacency list')
    if(not isinstance(y, tuple)):
        raise ValueError('If the first input is a tuple, the second input needs to be a tuple as well!')
    if(len(y) != 2):
        raise ValueError('If the second input is a tuple, it needs to contain exactly two elements, a node list and an adjacency list')
    x_nodes = x[0]
    x_adj   = x[1]
    y_nodes = y[0]
    y_adj   = y[1]
    return x_nodes, x_adj, y_nodes, y_adj


def outermost_right_leaves(list adj):
    """ Computes the outermost right leaves of a tree based on its adjacency
        list. The outermost right leaf of a tree is defined as recursively
        accessing the right-most child of a node until we hit a leaf.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    adj: An adjacency list representation of the tree, i.e. an array such that
         for every i, adj[i] is the list of child indices for node i.

    Returns: An array containing the outermost right leaf index for every node
             in the tree.
    """
    # the number of nodes in the tree
    cdef int m = len(adj)
    # the array into which we will write the outermost right leaves for each
    # node
    orl = np.full(m, -1, dtype=int)
    cdef long[:] orl_view = orl
    # a temporary variable for the current outermost right leaf
    cdef int r
    # iterate over all nodes to retrieve the respective outermost right leave
    cdef int i
    for i in range(m):
        # keep searching until we hit a node for which the outermost right
        # leaf is already defined or until we hit a leaf
        r = i
        while(True):
            # if r has no children, r is the outermost right leaf for i
            if(not adj[r]):
                orl_view[i] = r
                break
            # if the outermost right leaf for r is defined, that is also the
            # outermost right leaf for i
            if(orl_view[r] >= 0):
                orl_view[i] = orl_view[r]
                break
            # otherwise, continue searching
            r = adj[r][-1]
    return orl


def keyroots(long[:] orl):
    """ Computes the keyroots of a tree based on its outermost right leaf
        array. The keyroot for a node i is defined as the lowest k, such that
        orl[i] = orl[k].

    Args:
    orl: An outermost right leaf array as computed by the
         outermost_right_leaves function above.

    Returns: An array of keyroots in descending order.
    """
    # the number of nodes in the tree
    cdef int m = len(orl)
    # a temporary array to store the keyroots for each outermost right leaf
    kr = np.full(m, -1, dtype=int)
    cdef long[:] kr_view = kr
    # a variable to count the number of keyroots
    cdef int K = 0
    # iterate over all nodes
    cdef int i
    cdef int r
    for i in range(m):
        # check if a keyroot has been found for the current outermost
        # right leaf already
        r = orl[i]
        if(kr_view[r] < 0):
            # if not, store the new keyroot and increment the number of
            # keyroots
            kr_view[r] = i
            K += 1
    # in a next iteration, generate a new array which only contains
    # the defined keyroots
    keyroots = np.zeros(K, dtype=int)
    cdef long[:] keyroots_view = keyroots
    # insert and sort via insertionsort
    # counting index for the current keyroot
    cdef int k
    # insertion index for the current keyroot
    cdef int j
    # index for the current node
    i = 0
    for k in range(K):
        # iterate until the next keyroot is found
        while(kr_view[i] < 0):
            i += 1
        # decrement j until we found the location where the new keyroot should
        # be inserted
        j = k
        while(j > 0 and keyroots_view[j-1] < kr_view[i]):
            keyroots_view[j] = keyroots_view[j-1]
            j -= 1
        # insert the new keyroot
        keyroots_view[j] = kr_view[i]
        # increment i once more
        i += 1
    # return the resulting array
    return keyroots


@cython.boundscheck(False)
cdef void _ted_c(const long[:] x_orl, const long[:] x_kr, const long[:] y_orl, const long[:] y_kr, const double[:,:] Delta, double[:,:] D, double[:,:] D_tree) nogil:
    """ This method is internal and performs the actual tree edit distance
    computation for trees x and y in pure C.

    For details on the algorithm, please refer to the following tutorial:
    https://arxiv.org/abs/1805.06869

    Let m and n be the size of tree x and y respectively.

    Args:
    x_orl: the outermost right leaves for tree x (int array of length m).
    x_kr:  the keyroots for tree x in descending order (int array).
    y_orl: the outermost right leaves for tree y (int array of length n).
    y_kr:  the keyroots for tree y in descending order (int array).
    Delta: an (m+1) x (n+1) matrix, where Delta[i,j] for i < m, j < n is the
           cost of replacing x[i] with y[j], where Delta[i,n] is the cost of
           deleting x[i], and where Delta[m,j] is the cost of inserting y[j].
    D:     an empty (m+1) x (n+1) matrix used for temporary computations.
    D_tree: an empty m x n matrix. After this method has run, D_tree[i,j] will
            be the tree edit distance between the subtree rooted at i and the
            subtree rooted at j.
    """
    # the number of nodes in both trees
    cdef int m = len(x_orl)
    cdef int n = len(y_orl)
    # the number of keyroots in both trees
    cdef int K = len(x_kr)
    cdef int L = len(y_kr)

    # set up iteration variables
    # for the keyroots
    cdef int k
    cdef int l
    # for the nodes in the subtrees rooted at the keyroots
    cdef long i
    cdef long j
    # and temporary variables for the keyroots and the outermost right leaves
    cdef long i_0
    cdef long j_0
    cdef long i_max
    cdef long j_max

    # iterate over all pairwise combinations of keyroots
    for k in range(K):
        for l in range(L):
            # We consider now the subtree rooted at x_kr[k] versus the subtree
            # rooted at y_kr[l]. The forest edit distances between these
            # subtrees correspond exactly to the matrix block
            # D[x_kr[k]:x_orl[x_kr[k]]+1, y_kr[l]:y_orl[y_kr[l]]+1],
            # which we compute now.
            i_0 = x_kr[k]
            j_0 = y_kr[l]
            i_max = x_orl[i_0] + 1
            j_max = y_orl[j_0] + 1
            # first, initialize the last entry for the current subtree
            # computation
            D[i_max, j_max] = 0.
            # then, initialize the last column
            for i in range(i_max-1, i_0-1, -1):
                D[i, j_max] = Delta[i, n] + D[i+1, j_max]
            # then, initialize the last row
            for j in range(j_max-1, j_0-1, -1):
                D[i_max, j] = Delta[m, j] + D[i_max, j+1]
            # finally, compute the remaining forest edit distances
            for i in range(i_max-1, i_0-1, -1):
                for j in range(j_max-1, j_0-1, -1):
                    if(x_orl[i] == i_max-1 and y_orl[j] == j_max-1):
                        # if we consider a complete subtree, the forest edit
                        # distance D[i,j] is equal to the tree edit distance
                        # at that position and we can compute it via the
                        # standard edit distance recurrence
                        D[i,j] = min3(Delta[i,j] + D[i+1,j+1], # replacement
                                      Delta[i,n] + D[i+1,j], # deletion
                                      Delta[m,j] + D[i,j+1] # insertion
                                 )
                        # store the newly computed tree edit distance as well
                        D_tree[i,j] = D[i,j]
                    else:
                        # if we do _not_ consider a complete subtree, replacements
                        # are only possible between entire subtrees, which we have
                        # to consider in recurrence
                        D[i,j] = min3(D_tree[i,j] + D[x_orl[i]+1,y_orl[j]+1], # tree replacement
                                      Delta[i,n] + D[i+1,j], # deletion
                                      Delta[m,j] + D[i,j+1] # insertion
                                 )


cdef double min3(double a, double b, double c) nogil:
    """ Computes the minimum of three numbers, a, b, and c

    Args:
    a: the first number
    b: the second number
    c: the third number

    Returns: the minimum of a, b, and c.
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


# backtracing function

cdef double _BACKTRACE_TOL = 1E-5

def ted_backtrace(x_nodes, x_adj, y_nodes, y_adj, delta = None):
    """ Computes the tree edit distance between the trees x and y, each
    described by a list of nodes and an adjacency list adj, where adj[i]
    is a list of indices pointing to children of node i. This function
    returns an alignment representation of the distance.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    x_nodes: a list of nodes for tree x
             OR a tuple of the form (x_nodes, x_adj).
    x_adj:   an adjacency list for tree x
             OR a tuple of the form (y_nodes, y_adj).
    y_nodes: a list of nodes for tree y.
    y_adj:   an adjacency list for tree y.
    delta:   a function that takes two nodes as inputs and returns their
             pairwise distance, where delta(x, None) should be the cost of
             deleting x and delta(None, y) should be the cost of inserting y.
             If undefined, this method calls standard_ted_backtrace instead.

    Returns: a co-optimal alignment to edit x into y.
    """
    if(delta is None):
        return standard_ted_backtrace(x_nodes, x_adj, y_nodes, y_adj)

    if(isinstance(x_nodes, tuple)):
        x_nodes, x_adj, y_nodes, y_adj = extract_from_tuple_input(x_nodes, x_adj)

    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)
    # if both trees are empty, the distance is necessarily zero.
    if(m == 0 and n == 0):
        return 0.
    # An array to store all edit costs for replacements, deletions, and
    # insertions
    Delta = np.zeros((m+1, n+1))
    cdef double[:,:] Delta_view = Delta
    # First, compute all pairwise replacement costs
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            Delta_view[i,j] = delta(x_nodes[i], y_nodes[j])

    # Then, compute the deletion and insertion costs
    for i in range(m):
        Delta_view[i,n] = delta(x_nodes[i], None)
    for j in range(n):
        Delta_view[m,j] = delta(None, y_nodes[j])

    # if either tree is empty, we can only delete/insert all nodes in the
    # non-empty tree.
    if(m == 0):
        return np.sum(Delta[0,:])
    if(n == 0):
        return np.sum(Delta[:,0])

    # Compute the keyroots and outermost right leaves for both trees.
    x_orl = outermost_right_leaves(x_adj)
    x_kr  = keyroots(x_orl)
    y_orl = outermost_right_leaves(y_adj)
    y_kr  = keyroots(y_orl)

    # Compute the actual tree edit distance
    D = np.zeros((m+1,n+1))
    D_tree = np.zeros((m,n))
    _ted_c(x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree)

    # construct views for the matrices Delta, D, and D_tree
    cdef double[:,:] D_view = D
    cdef double[:,:] D_tree_view = D_tree

    # initialize the alignment
    ali = Alignment()

    # start the recursive backtrace computation
    _ted_backtrace(x_orl, y_orl, Delta_view, D_view, D_tree_view, ali, 0, 0)
    return ali

def _ted_backtrace(const long[:] x_orl, const long[:] y_orl, double[:,:] Delta, double[:,:] D, const double[:,:] D_tree, ali, int k, int l):
    """ Performs the backtracing for the subtree rooted at k in x versus the
        subtree rooted at l in y.
    """
    # recompute the dynamic programming matrix for the current subtree
    # combination
    cdef int m = len(x_orl)
    cdef int n = len(y_orl)
    cdef int i_max = x_orl[k] + 1
    cdef int j_max = y_orl[l] + 1
    cdef int i
    cdef int j
    if(k > 0 or l > 0):
        # note that D[i_max, j_max] is already correctly initialized
        # initialize the last column
        for i in range(i_max-1, k-1, -1):
            D[i, j_max] = 1 + D[i+1, j_max]
        # then, initialize the last row
        for j in range(j_max-1, l-1, -1):
            D[i_max, j] = 1 + D[i_max, j+1]
        # finally, compute the remaining forest edit distances
        for i in range(i_max-1, k-1, -1):
            for j in range(j_max-1, l-1, -1):
                if(x_orl[i] == x_orl[k] and y_orl[j] == y_orl[l]):
                    # if we consider a complete subtree, we can re-use
                    # the tree edit distance values we computed in the
                    # forward pass
                    D[i,j] = D_tree[i,j] + D[i_max, j_max]
                else:
                    # if we do _not_ consider a complete subtree,
                    # replacements are only possible between entire
                    # subtrees, which we have to consider in
                    # recurrence
                    D[i,j] = min3(D_tree[i,j] + D[x_orl[i]+1,y_orl[j]+1], # tree replacement
                                  Delta[i,n] + D[i+1,j], # deletion
                                  Delta[m,j] + D[i,j+1]  # insertion
                             )
    # now, start the backtracing for the current subtree combination
    i = k
    j = l
    while(i < i_max and j < j_max):
        # check whether a deletion is co-optimal
        if(D[i, j] + _BACKTRACE_TOL > Delta[i, n] + D[i+1, j]):
            # if so, append a deletion operation, increment i, and continue
            ali.append_tuple(i, -1)
            i += 1
            continue
        # check whether an insertion is co-optimal
        if(D[i, j] + _BACKTRACE_TOL > Delta[m, j] + D[i, j+1]):
            # if so, append an insertion operation, increment j, and continue
            ali.append_tuple(-1, j)
            j += 1
            continue
        # check wehther replacement is co-optimal. In this case, we need to
        # consider two cases
        if(x_orl[i] == x_orl[k] and y_orl[j] == y_orl[l]):
            # If we are at the root of postfix-subtrees for subtree k and l,
            # we consider the standard replacement case
            if(D[i,j] + _BACKTRACE_TOL > Delta[i,j] + D[i+1,j+1]):
                # append a replacement operation, increment i and j, and
                # continue
                ali.append_tuple(i, j)
                i += 1
                j += 1
                continue
        else:
            if(D[i, j] + _BACKTRACE_TOL > D_tree[i,j] + D[x_orl[i]+1,y_orl[j]+1]):
                # Otherwise, we consider the case where we replace the entire
                # subtree rooted at i with the entire subtree rooted at j.
                # For this case, we call the backtracing recursively
                _ted_backtrace(x_orl, y_orl, Delta, D, D_tree, ali, i, j)
                i = x_orl[i]+1
                j = y_orl[j]+1
                continue
        # if we got here, nothing is co-optimal, which is an error
        raise ValueError('Internal error: No option is co-optimal.')
    # delete and insert any remaining nodes
    while(i < i_max):
        ali.append_tuple(i, -1)
        i += 1
    while(j < j_max):
        ali.append_tuple(-1, j)
        j += 1

# the standard edit distance with kronecker distance

def standard_ted(x_nodes, x_adj, y_nodes = None, y_adj = None):
    """ Computes the standard tree edit distance between the trees x and y,
    each described by a list of nodes and an adjacency list adj, where adj[i]
    is a list of indices pointing to children of node i.

    The 'standard' refers to the fact that we use the kronecker distance
    as delta, i.e. this call computes the same as

    ted(x_nodes, x_adj, y_nodes, y_adj, kronecker_distance) where

    kronecker_distance(x, y) = 1 if x != y and 0 if x = y.

    However, this implementation here is notably faster because we can apply
    integer arithmetic.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    x_nodes: a list of nodes for tree x
             OR a tuple of the form (x_nodes, x_adj).
    x_adj:   an adjacency list for tree x
             OR a tuple of the form (y_nodes, y_adj).
    y_nodes: a list of nodes for tree y.
    y_adj:   an adjacency list for tree y.

    Returns: the standard tree edit distance between x and y as an integer.
    """

    if(isinstance(x_nodes, tuple)):
        x_nodes, x_adj, y_nodes, y_adj = extract_from_tuple_input(x_nodes, x_adj)

    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)
    # if the left tree is empty, the standard edit distance is n, and vice
    # versa
    if(m == 0):
        return n
    if(n == 0):
        return m
    # An array to store which pairs of symbols in x and y are equal
    Delta = np.zeros((m, n), dtype=int)
    cdef long[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            if(x_nodes[i] != y_nodes[j]):
                Delta_view[i,j] = 1

    # Compute the keyroots and outermost right leaves for both trees.
    x_orl = outermost_right_leaves(x_adj)
    x_kr  = keyroots(x_orl)
    y_orl = outermost_right_leaves(y_adj)
    y_kr  = keyroots(y_orl)

    # Finally, compute the actual tree edit distance
    D_forest = np.zeros((m+1,n+1), dtype=int)
    D_tree = np.zeros((m,n), dtype=int)
    _std_ted_c(x_orl, x_kr, y_orl, y_kr, Delta, D_forest, D_tree)
    return D_tree[0,0]

@cython.boundscheck(False)
cdef void _std_ted_c(const long[:] x_orl, const long[:] x_kr, const long[:] y_orl, const long[:] y_kr, const long[:,:] Delta, long[:,:] D, long[:,:] D_tree) nogil:
    """ This method is internal and performs the actual standard tree edit
    distance computation for trees x and y in pure C.

    For details on the algorithm, please refer to the following tutorial:
    https://arxiv.org/abs/1805.06869

    Let m and n be the size of tree x and y respectively.

    Args:
    x_orl: the outermost right leaves for tree x (int array of length m).
    x_kr:  the keyroots for tree x in descending order (int array).
    y_orl: the outermost right leaves for tree y (int array of length n).
    y_kr:  the keyroots for tree y in descending order (int array).
    Delta: an m x n int matrix, where Delta[i,j] is 1 if x[i] != y[j] and 0
           if x[i] == y[j]
    D:     an empty (m+1) x (n+1) matrix used for temporary computations.
    D_tree: an empty m x n matrix. After this method has run, D_tree[i,j] will
            be the tree edit distance between the subtree rooted at i and the
            subtree rooted at j.
    """
    # the number of nodes in both trees
    cdef int m = len(x_orl)
    cdef int n = len(y_orl)
    # the number of keyroots in both trees
    cdef int K = len(x_kr)
    cdef int L = len(y_kr)

    # set up iteration variables
    # for the keyroots
    cdef int k
    cdef int l
    # for the nodes in the subtrees rooted at the keyroots
    cdef long i
    cdef long j
    # and temporary variables for the keyroots and the outermost right leaves
    cdef long i_0
    cdef long j_0
    cdef long i_max
    cdef long j_max

    # iterate over all pairwise combinations of keyroots
    for k in range(K):
        for l in range(L):
            # We consider now the subtree rooted at x_kr[k] versus the subtree
            # rooted at y_kr[l]. The forest edit distances between these
            # subtrees correspond exactly to the matrix block
            # D[x_kr[k]:x_orl[x_kr[k]]+1, y_kr[l]:y_orl[y_kr[l]]+1],
            # which we compute now.
            i_0 = x_kr[k]
            j_0 = y_kr[l]
            i_max = x_orl[i_0] + 1
            j_max = y_orl[j_0] + 1
            # first, initialize the last entry for the current subtree
            # computation
            D[i_max, j_max] = 0
            # then, initialize the last column
            for i in range(i_max-1, i_0-1, -1):
                D[i, j_max] = 1 + D[i+1, j_max]
            # then, initialize the last row
            for j in range(j_max-1, j_0-1, -1):
                D[i_max, j] = 1 + D[i_max, j+1]
            # finally, compute the remaining forest edit distances
            for i in range(i_max-1, i_0-1, -1):
                for j in range(j_max-1, j_0-1, -1):
                    if(x_orl[i] == x_orl[i_0] and y_orl[j] == y_orl[j_0]):
                        # if we consider a complete subtree, the forest edit
                        # distance D[i,j] is equal to the tree edit distance
                        # at that position and we can compute it via the
                        # standard edit distance recurrence
                        D[i,j] = min3_int(Delta[i,j] + D[i+1,j+1], # replacement
                                      1 + D[i+1,j], # deletion
                                      1 + D[i,j+1] # insertion
                                 )
                        # store the newly computed tree edit distance as well
                        D_tree[i,j] = D[i,j]
                    else:
                        # if we do _not_ consider a complete subtree, replacements
                        # are only possible between entire subtrees, which we have
                        # to consider in recurrence
                        D[i,j] = min3_int(D_tree[i,j] + D[x_orl[i]+1,y_orl[j]+1], # tree replacement
                                      1 + D[i+1,j], # deletion
                                      1 + D[i,j+1] # insertion
                                 )

cdef long min3_int(long a, long b, long c) nogil:
    """ Computes the minimum of three numbers, a, b, and c

    Args:
    a: the first number
    b: the second number
    c: the third number

    Returns: the minimum of a, b, and c.
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


# backtracing function

def standard_ted_backtrace(x_nodes, x_adj, y_nodes, y_adj):
    """ Computes the standard tree edit distance between the trees x and y,
    each described by a list of nodes and an adjacency list adj, where adj[i]
    is a list of indices pointing to children of node i. This function
    returns an alignment representation of the distance.

    The 'standard' refers to the fact that we use the kronecker distance
    as delta, i.e. this call computes the same as

    ted(x_nodes, x_adj, y_nodes, y_adj, kronecker_distance) where

    kronecker_distance(x, y) = 1 if x != y and 0 if x = y.

    However, this implementation here is notably faster because we can apply
    integer arithmetic.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    x_nodes: a list of nodes for tree x
             OR a tuple of the form (x_nodes, x_adj).
    x_adj:   an adjacency list for tree x
             OR a tuple of the form (y_nodes, y_adj).
    y_nodes: a list of nodes for tree y.
    y_adj:   an adjacency list for tree y.

    Returns: a co-optimal alignment to edit x into y.
    """

    if(isinstance(x_nodes, tuple)):
        x_nodes, x_adj, y_nodes, y_adj = extract_from_tuple_input(x_nodes, x_adj)

    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)
    # if the left tree is empty, the standard edit distance is n, and vice
    # versa
    if(m == 0):
        return n
    if(n == 0):
        return m
    # An array to store which pairs of symbols in x and y are equal
    Delta = np.zeros((m, n), dtype=int)
    cdef long[:,:] Delta_view = Delta
    cdef int i
    cdef int j
    for i in range(m):
        for j in range(n):
            if(x_nodes[i] != y_nodes[j]):
                Delta_view[i,j] = 1

    # Compute the keyroots and outermost right leaves for both trees.
    cdef long[:] x_orl = outermost_right_leaves(x_adj)
    cdef long[:] x_kr  = keyroots(x_orl)
    cdef long[:] y_orl = outermost_right_leaves(y_adj)
    cdef long[:] y_kr  = keyroots(y_orl)

    # compute the actual tree edit distance
    D = np.zeros((m+1,n+1), dtype=int)
    D_tree = np.zeros((m,n), dtype=int)
    # construct views for the matrices Delta, D, and D_tree
    cdef long[:,:] D_view = D
    cdef long[:,:] D_tree_view = D_tree
    _std_ted_c(x_orl, x_kr, y_orl, y_kr, Delta_view, D_view, D_tree_view)

    # initialize the alignment
    ali = Alignment()
    # start backtracing recursively
    _standard_ted_backtrace(x_orl, y_orl, Delta_view, D_view, D_tree_view, ali, 0, 0)
    return ali

def _standard_ted_backtrace(const long[:] x_orl, const long[:] y_orl, long[:,:] Delta, long[:,:] D, const long[:,:] D_tree, ali, int k, int l):
    """ Performs the backtracing for the subtree rooted at k in x versus the
        subtree rooted at l in y.
    """
    # recompute the dynamic programming matrix for the current subtree
    # combination
    cdef int i_max = x_orl[k] + 1
    cdef int j_max = y_orl[l] + 1
    cdef int i = 0
    cdef int j = 0
    if(k > 0 or l > 0):
        # if we are either at k > 0 or l > 0, the first action is a replacement
        ali.append_tuple(k, l)
        # note that D[i_max, j_max] is already correctly initialized
        # initialize the last column
        for i in range(i_max-1, k, -1):
            D[i, j_max] = 1 + D[i+1, j_max]
        # then, initialize the last row
        for j in range(j_max-1, l, -1):
            D[i_max, j] = 1 + D[i_max, j+1]
        # finally, compute the remaining forest edit distances
        for i in range(i_max-1, k, -1):
            for j in range(j_max-1, l, -1):
                if(x_orl[i] == x_orl[k] and y_orl[j] == y_orl[l]):
                    # if we consider a complete subtree, we can re-use
                    # the tree edit distance values we computed in the
                    # forward pass
                    D[i,j] = D_tree[i,j] + D[i_max, j_max]
                else:
                    # if we do _not_ consider a complete subtree,
                    # replacements are only possible between entire
                    # subtrees, which we have to consider in
                    # recurrence
                    D[i,j] = min3_int(D_tree[i,j] + D[x_orl[i]+1,y_orl[j]+1], # tree replacement
                                  1 + D[i+1,j], # deletion
                                  1 + D[i,j+1] # insertion
                             )
        i = k + 1
        j = l + 1
    # now, start the backtracing for the current subtree combination
    while(i < i_max and j < j_max):
        # check whether a deletion is co-optimal
        if(D[i, j] == 1 + D[i+1, j]):
            # if so, append a deletion operation, increment i, and continue
            ali.append_tuple(i, -1)
            i += 1
            continue
        # check whether an insertion is co-optimal
        if(D[i, j] == 1 + D[i, j+1]):
            # if so, append an insertion operation, increment j, and continue
            ali.append_tuple(-1, j)
            j += 1
            continue
        # check wehther replacement is co-optimal. In this case, we need to
        # consider two cases
        if(x_orl[i] == x_orl[k] and y_orl[j] == y_orl[l]):
            # If we are at the root of postfix-subtrees for subtree k and l,
            # we consider the standard replacement case
            if(D[i,j] == Delta[i,j] + D[i+1,j+1]):
                # append a replacement operation, increment i and j, and
                # continue
                ali.append_tuple(i, j)
                i += 1
                j += 1
                continue
        else:
            if(D[i,j] == D_tree[i,j] + D[x_orl[i]+1,y_orl[j]+1]):
                # Otherwise, we consider the case where we replace the entire
                # subtree rooted at i with the entire subtree rooted at j.
                # For this case, we call the backtracing recursively
                _standard_ted_backtrace(x_orl, y_orl, Delta, D, D_tree, ali, i, j)
                i = x_orl[i]+1
                j = y_orl[j]+1
                continue
        # if we got here, nothing is co-optimal, which is an error
        raise ValueError('Internal error: No option is co-optimal.')
    # delete and insert any remaining nodes
    while(i < i_max):
        ali.append_tuple(i, -1)
        i += 1
    while(j < j_max):
        ali.append_tuple(-1, j)
        j += 1
