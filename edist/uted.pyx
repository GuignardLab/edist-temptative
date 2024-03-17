#!python
#cython: language_level=3
"""
Implements the constrained unordered tree edit distance of Zhang (1996) and its
backtracing in cython.

"""
# Copyright (C) 2021
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

from edist.ted import extract_from_tuple_input
import numpy as np
from scipy.optimize import linear_sum_assignment
from edist.alignment import Alignment
cimport cython

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.2'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'


cdef extern from "math.h":
    float INFINITY

###################################
# Edit Distance with Custom Delta #
###################################

def uted(x_nodes, x_adj, y_nodes = None, y_adj = None, delta = None):
    """ Computes the constrained, unordered tree edit distance between the
     trees x and y, each described by a list of nodes and an adjacency list
    adj, where adj[i] is a list of indices pointing to children of node i.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Parameters
    ----------
    x_nodes: list or tuple
        a list of nodes for tree x OR a tuple of the form (x_nodes, x_adj).
    x_adj: list or tuple
        an adjacency list for tree x OR a tuple of the form (y_nodes, y_adj).
    y_nodes: list (default = x_adj[0])
        a list of nodes for tree y.
    y_adj: list (default = x_adj[1])
        an adjacency list for tree y.
    delta: function (default = None)
        a function that takes two nodes as inputs and returns their pairwise
        distance, where delta(x, None) should be the cost of deleting x and
        delta(None, y) should be the cost of inserting y. If undefined, this
        method uses unit costs.

    Returns
    -------
    d: float
        the tree edit distance between x and y according to delta.

    """
    if isinstance(x_nodes, tuple):
        x_nodes, x_adj, y_nodes, y_adj = extract_from_tuple_input(x_nodes, x_adj)

    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)

    # if either tree is empty, we can only delete/insert all nodes in the
    # non-empty tree.
    cdef double d = 0
    cdef int i
    cdef int j
    if m == 0:
        if delta is None:
            return n
        else:
            for j in range(n):
                d += delta(None, y_nodes[j])
            return d
    if n == 0:
        if delta is None:
            return m
        else:
            for i in range(m):
                d += delta(x_nodes[i], None)
            return d

    # Set up an array to store edit costs for replacements,
    # deletions, and insertions
    Delta = np.ones((m+1, n+1))
    cdef double[:,:] Delta_view = Delta

    if delta is None:
        for i in range(m):
            for j in range(n):
                if x_nodes[i] == y_nodes[j]:
                    Delta_view[i, j] = 0.
    else:
        # First, compute all pairwise replacement costs
        for i in range(m):
            for j in range(n):
                Delta_view[i,j] = delta(x_nodes[i], y_nodes[j])
        # Then, compute the deletion and insertion costs
        for i in range(m):
            Delta_view[i,n] = delta(x_nodes[i], None)
        for j in range(n):
            Delta_view[m,j] = delta(None, y_nodes[j])

    # compute the actual tree edit distance
    D_forest, D_tree = _uted(x_nodes, x_adj, y_nodes, y_adj, Delta)

    return D_tree[0,0]

def _uted(x_nodes, x_adj, y_nodes, y_adj, Delta):
    """ Internal function; call uted instead. """
    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)

    # convert adjacency lists to array form
    A_x, deg_x = adjmat_(x_adj)
    A_y, deg_y = adjmat_(y_adj)

    # set up temporary matrices for the munkres algorithm
    cdef int max_deg = A_x.shape[1] + A_y.shape[1]
    C = np.zeros((max_deg, max_deg))
    Stars  = np.zeros((max_deg, max_deg), dtype=np.intc)
    Primes = np.zeros((max_deg, max_deg), dtype=np.intc)
    row_covers = np.zeros(max_deg, dtype=np.intc)
    col_covers = np.zeros(max_deg, dtype=np.intc)
    path = np.zeros((max_deg, 2), dtype=int)
    pi = np.zeros(max_deg, dtype=int)

    # initialize dynamic programming matrices for forest and tree edit distance
    D_forest = np.zeros((m+1,n+1))
    D_tree = np.zeros((m+1,n+1))
    
    # call the c routine
    uted_c_(A_x, deg_x, A_y, deg_y, Delta, D_forest, D_tree, C, Stars, Primes, row_covers, col_covers, path, pi)

    return D_forest, D_tree

def adjmat_(adj):
    """ Converts an adjacency list into an int array """
    cdef int m = len(adj)
    degs = np.zeros(m, dtype=int)
    cdef long[:] deg_view = degs
    cdef int i
    for i in range(m):
        deg_view[i] = len(adj[i])

    cdef long max_deg = np.max(degs)
    Adj = np.zeros((m, max_deg), dtype=int)
    cdef long[:, :] Adj_view = Adj

    cdef int k
    for i in range(m):
        for k in range(deg_view[i]):
            Adj_view[i, k] = adj[i][k]

    return Adj, degs


@cython.boundscheck(False)
cdef void uted_c_(const long[:,:] A_x, const long[:] deg_x, const long[:,:] A_y, const long[:] deg_y, 
    const double[:,:] Delta, double[:,:] D_forest, double[:,:] D_tree,
    double[:, :] C, int[:, :] Stars, int[:, :] Primes,
    int[:] row_covers, int[:] col_covers, long[:, :] path, long[:] pi) noexcept nogil:
    """ This method is internal and performs the actual tree edit distance
    computation for trees x and y in pure C.

    For details on the algorithm, please refer to the paper
    'A Constrained Edit Distance Between Unordered Labeled Trees'
    by Zhang (1996).

    """
    # the number of nodes in both trees
    cdef int m = A_x.shape[0]
    cdef int n = A_y.shape[0]

    # for the nodes in x and y
    cdef int i
    cdef int j
    # for the children of x[i] and y[j]
    cdef int k
    cdef int l
    # for the number of children of i and j
    cdef int m_i
    cdef int n_j
    # and for the child indices
    cdef int i_k = 0
    cdef int j_l = 0
    # and for the deletion, insertion, and replacement cost
    cdef double del_cost
    cdef double ins_cost
    cdef double rep_cost

    cdef double tmp_cost

    for i in range(m-1, -1, -1):
        # Compute the subforest deletion cost for i, i.e. the cost
        # for deleting all of i's child subtrees
        for k in range(deg_x[i]):
            i_k = A_x[i, k]
            D_forest[i, n] += D_tree[i_k, n]
        # Deleting the tree rooted at i means deleting node i and all its
        # children
        D_tree[i, n] = Delta[i, n] + D_forest[i, n]

    for j in range(n-1, -1, -1):
        # Compute the subforest insertion cost for j, i.e. the cost
        # for inserting all of j's child subtrees
        for l in range(deg_y[j]):
            j_l = A_y[j, l]
            D_forest[m, j] += D_tree[m, j_l]
        # Inserting the tree rooted at j means inserting node j and all its
        # children
        D_tree[m, j] = Delta[m, j] + D_forest[m, j]

    # now, start the actual recursion
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            m_i = deg_x[i]
            n_j = deg_y[j]
            # First, we compute the forest edit distance, i.e. the cost for
            # editing all children of i into all children of j.

            # We consider first the special case that either i or
            # j have no children. Then, the computation is really
            # simple because we can only delete/insert
            if m_i == 0:
                D_forest[i, j] = D_forest[m, j]
            elif n_j == 0:
                D_forest[i, j] = D_forest[i, n]
            else:
                # if both nodes have children, perform the actual computation.
                # For that, we have three options.
                # First, we could delete all children of i except for a single
                # subtree
                del_cost = INFINITY
                for k in range(m_i):
                    i_k = A_x[i, k]
                    # accordingly, we need to consider the cost of editing
                    # the children of node i_k with the children of j,
                    # plus the cost of deleting all other children of i.
                    tmp_cost = D_forest[i_k, j] + D_forest[i, n] - D_forest[i_k, n]
                    if tmp_cost < del_cost:
                        del_cost = tmp_cost
                # Second, we could insert all children of j except for a single
                # subtree
                ins_cost = INFINITY
                for l in range(n_j):
                    j_l = A_y[j, l]
                    # accordingly, we need to consider the cost of editing
                    # the children of node i to the children of j_l,
                    # plus the cost of inserting all other children of j.
                    tmp_cost = D_forest[i, j_l] + D_forest[m, j] - D_forest[m, j_l]
                    if tmp_cost < ins_cost:
                        ins_cost = tmp_cost
                # Third, we could replace, meaning that we optimally match all
                # children of i to all children of j.

                # if there is only one child in one of the forests, we can
                # solve this with a simplified algorithm
                if m_i == 1:
                    # we obtain the total cost by computing the
                    # cost of inserting all children of j,
                    # minus the cost of inserting one child,
                    # but replacing it with i_k
                    rep_cost = INFINITY
                    i_k = A_x[i, 0]
                    for l in range(n_j):
                        j_l = A_y[j, l]
                        tmp_cost = D_tree[i_k, j_l] + D_forest[m, j] - D_tree[m, j_l]
                        if tmp_cost < rep_cost:
                            rep_cost = tmp_cost
                elif n_j == 1:
                    # we obtain the total cost by computing the
                    # cost of deleting all children of i,
                    # minus the cost of deleting one child,
                    # but replacing it with j_k
                    rep_cost = INFINITY
                    j_l = A_y[j, 0]
                    for k in range(m_i):
                        i_k = A_x[i, k]
                        tmp_cost = D_tree[i_k, j_l] + D_forest[i, n] - D_tree[i_k, n]
                        if tmp_cost < rep_cost:
                            rep_cost = tmp_cost
                else:
                    # if there is more than one child on both sides, we use
                    # the Munkres/Hungarian algorithm.

                    # prepare a cost matrix for the Hungarian algorithm
                    for k in range(m_i):
                        i_k = A_x[i, k]
                        for l in range(n_j):
                            j_l = A_y[j, l]
                            # matching ci with cj means editing the ci'th
                            # child of i to the cj'th child of j.
                            C[k, l] = D_tree[i_k, j_l]
                    C[:m_i, n_j:m_i+n_j] = INFINITY
                    for k in range(m_i):
                        # matching c with n_j + c means deleting the
                        # c'th child of i
                        i_k = A_x[i, k]
                        C[k, n_j + k] = D_tree[i_k, n]
                    C[m_i:m_i+n_j, :n_j] = INFINITY
                    for l in range(n_j):
                        # matching m_i + c with c means inserting the
                        # c'th child of j
                        j_l = A_y[j, l]
                        C[m_i + l, l] = D_tree[m, j_l]
                    C[m_i:m_i+n_j, n_j:m_i+n_j] = 0.
                    # print('i = %d, j = %d' % (i, j))
                    # print('C = %s' % str(np.asarray(C[:m_i+n_j,:m_i+n_j])))
                    # solve the linear sum assignment problem for C. The resulting
                    # minimum cost is our replacement cost
                    # print('C before munkres for (%d, %d):\n%s' % (i, j, str(np.asarray(C))))
                    munkres_(C[:m_i+n_j,:m_i+n_j], Stars[:m_i+n_j,:m_i+n_j], Primes[:m_i+n_j,:m_i+n_j],
                    row_covers[:m_i+n_j], col_covers[:m_i+n_j], path[:m_i+n_j, :], pi[:m_i+n_j])
                    # print('pi after munkres: %s' % str(np.asarray(pi)))
                    # reconstruct the cost of the assignment from pi
                    rep_cost = 0.
                    for k in range(m_i):
                        i_k = A_x[i, k]
                        if pi[k] >= n_j:
                            # if pi[k] >= n_j, tree i_k should be deleted
                            rep_cost += D_tree[i_k, n]
                            # print('delete %d for %g' % (i_k, D_tree[i_k, n]))
                        else:
                            # otherwise we replace i_k with j_pi[k]
                            j_l = A_y[j, pi[k]]
                            rep_cost += D_tree[i_k, j_l]
                            # print('replace %d with %d for %g' % (i_k, j_l, D_tree[i_k, j_l]))
                    for l in range(n_j):
                        j_l = A_y[j, l]
                        if pi[m_i + l] < n_j:
                            # if pi[m_i + l] < n_j, tree j_l should be inserted
                            rep_cost += D_tree[m, j_l]
                            # print('insert %d for %g' % (j_l, D_tree[m, j_l]))
                    # print('cost of pi: %g' % rep_cost)
                # compute minimum across deletion, insertion, and replacement
                # print('del_cost = %g, ins_cost = %g, rep_cost = %g' % (del_cost, ins_cost, rep_cost))
                D_forest[i, j] = min3(del_cost, ins_cost, rep_cost)

            # next, compute the unordered tree edit distance between the
            # subtrees rooted at i and j.
            # Again, we have three options.
            # First, we could delete node i and all subtrees except a
            # single one.
            if m_i == 0:
                # if i has no children, we delete i and insert
                # everything in j
                del_cost = Delta[i, n] + D_tree[m, j]
            else:
                del_cost = INFINITY
                for k in range(m_i):
                    i_k = A_x[i, k]
                    # accordingly, we need to consider the cost of editing
                    # tree i_k into tree j plus the cost of deleting all other
                    # children of i.
                    tmp_cost = D_tree[i_k, j] + D_tree[i, n] - D_tree[i_k, n]
                    if tmp_cost < del_cost:
                        del_cost = tmp_cost
            # Second, we could insert node j and all children of j except
            # for a single one.
            if n_j == 0:
                # if j has no children, we insert j and delete
                # everything in i
                ins_cost = Delta[m, j] + D_tree[i, n]
            else:
                ins_cost = INFINITY
                for l in range(n_j):
                    j_l = A_y[j, l]
                    # accordingly, we need to consider the cost of editing
                    # tree i into tree j_l plus the cost o inserting all other
                    # children of j.
                    tmp_cost = D_tree[i, j_l] + D_tree[m, j] - D_tree[m, j_l]
                    if tmp_cost < ins_cost:
                        ins_cost = tmp_cost
            # Third, we could replace, meaning that we edit node i into
            # node j and all children of i into all children of j
            rep_cost = Delta[i, j] + D_forest[i, j]
            # compute minimum across deletion, insertion, and replacement
            D_tree[i, j] = min3(del_cost, ins_cost, rep_cost)

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

###############################
# Munkres/Hungarian algorithm #
###############################

@cython.boundscheck(False)
cdef void munkres_(double[:, :] C, int[:, :] Stars, int[:, :] Primes,
    int[:] row_covers, int[:] col_covers, long[:, :] path, long[:] pi) noexcept nogil:
    """ Implements the munkres algorithm to find a minimal matching
    for the cost matrix C. All other inputs are auxiliary matrices
    and vectors for computation.

    This implementation strictly follows the description of the
    algorithm given by Bourgeois and Lasalle (1971) in their paper
    "An Extension of the Munkres Algorithm for the Assignment Problem
    to Rectangular Matrices".

    The final assignment is stored in pi afterwards.
    """
    cdef int m = C.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int l = 0
    cdef double h
    cdef int t
    cdef int T
    cdef int found_zero

    # initialization: we subtract the row and column minimum
    for i in range(m):
        h = INFINITY
        for j in range(m):
            if C[i, j] < h:
                h = C[i, j]
        for j in range(m):
            C[i, j] -= h
    for j in range(m):
        h = INFINITY
        for i in range(m):
            if C[i, j] < h:
                h = C[i, j]
        for i in range(m):
            C[i, j] -= h
    # print('C after initialization:\n%s' % str(np.asarray(C)))
    # initialize all auxiliary matrices as false
    Stars[:, :]   = 0
    Primes[:, :]  = 0
    row_covers[:] = 0
    col_covers[:] = 0
    # Step 1: inspect all zeros in the matrix. If there is
    # no starred zero in its row or column, star it.
    for i in range(m):
        for j in range(m):
            if C[i, j] < 1E-5 and row_covers[i] <= 0 and col_covers[j] <= 0:
                Stars[i, j]   = 1
                row_covers[i] = 1
                col_covers[j]  = 1
    # print('Stars after initialization\n%s' % str(np.asarray(Stars)))
    while True:
        # uncover all rows but keep columns covered
        row_covers[:] = False
        # Step 2: uncover all rows, but keep columns
        # with a starred zero covered. If that covers
        # all columns, we are done.
        found_zero = 0
        for j in range(m):
            if col_covers[j] <= 0:
                found_zero = 1
                break
        # print('checking step 2: %d' % found_zero)
        if found_zero <= 0:
            # recover the assignment
            for i in range(m):
                for j in range(m):
                    if Stars[i, j] > 0:
                        pi[i] = j
                        break
            # end computation
            return
        # Step 3: Prime non-covered zeros and change coverings
        while True:
            # find an uncovered zero
            found_zero = 0
            # print('row covers in step 3: %s' % str(np.asarray(row_covers)))
            # print('col covers in step 3: %s' % str(np.asarray(col_covers)))
            for i in range(m):
                if row_covers[i] > 0:
                    continue
                for j in range(m):
                    if col_covers[j] > 0:
                        continue
                    if C[i, j] < 1E-5:
                        found_zero = 1
                        break
                if found_zero > 0:
                    break
            # print('found zero in step 3: %d' % found_zero)
            if found_zero > 0:
                Primes[i, j] = 1
                # print('primed (%d, %d) in step 3' % (i, j))
                # search for a zero in the same row
                for l in range(m):
                    if Stars[i, l] > 0:
                        break
                if Stars[i, l]:
                    # print('covered row %d and uncovered column %d in step 3' % (i, l))
                    # if there is a starred zero in the same row
                    # as a primed zero, cover the row and uncover
                    # the column of the starred zero
                    row_covers[i] = 1
                    col_covers[l] = 0
                else:
                    # break and go to step 4
                    break
            else:
                # step 5
                # find non-covered minimum
                h = INFINITY
                for j in range(m):
                    if col_covers[j] > 0:
                        continue
                    for i in range(m):
                        if row_covers[i] > 0:
                            continue
                        if C[i, j] < h:
                            h = C[i, j]
                # add it to covered rows
                for i in range(m):
                    if row_covers[i] > 0:
                        for j in range(m):
                            C[i, j] += h
                # and uncovered columns
                for j in range(m):
                    if col_covers[j] > 0:
                        continue
                    for i in range(m):
                        C[i, j] -= h
                # print('new C after step 5\n%s' % str(np.asarray(C)))
                # conclude step 5, return to step 3

        # step 3 loop is completed, start step 4

        # step 4. We should now find a sequence of
        # starred and primed zeros by alternating
        # rows and columns.
        k = i
        l = j
        # print('Found prime at (%d, %d) in step 4' % (k, l))
        path[0, 0] = k
        path[0, 1] = l
        T = 1
        # continue the path as far as possible
        while True:
            # find the next star in the same column
            for k in range(m):
                if Stars[k, l] > 0:
                    break
            if Stars[k, l] <= 0:
                # if there is no star, the path is complete
                break
            # print('Found star at (%d, %d) in step 4' % (k, l))
            # if there is a star, remove it
            Stars[k, l] = 0
            # then, find the next prime in the same column.
            # This prime must exist
            for l in range(m):
                if Primes[k, l] > 0:
                    break
            # print('Found prime at (%d, %d) in step 4' % (k, l))
            # un-prime it and add it to the path
            Primes[k, l] = 0
            path[T, 0] = k
            path[T, 1] = l
            T += 1
        # star all locations on the path
        for t in range(T):
            Stars[path[t, 0], path[t, 1]] = 1
        # remove all primes
        Primes[:, :] = 0
        # update all column covers
        col_covers[:] = 0
        for j in range(m):
            for i in range(m):
                if Stars[i, j] > 0:
                    col_covers[j] = 1
                    break
        # step 4 complete, return to step 2
        # print('new stars after step 4\n%s' % str(np.asarray(Stars)))
        # print('new col covers after step 4: %s' % str(np.asarray(col_covers)))

def munkres(C):
    """ This calls the Munkres algorithm to find a minimal
    matching for the given cost matrix C. Note that this function
    is more for debugging purposes. If you want to call this
    algorithm from Python, you are better served by calling
    scipy.optimize.linear_sum_assignment.

    Parameters
    ----------
    C: ndarray
        An m x m cost matrix.

    Returns
    -------
    pi: ndarray
        An m-element array where pi[i] is the index to which i is
        assigned.

    """
    # set up all temporary variables
    C = np.copy(C)
    m = C.shape[0]
    Stars  = np.zeros((m, m), dtype=np.intc)
    Primes = np.zeros((m, m), dtype=np.intc)
    row_covers = np.zeros(m, dtype=np.intc)
    col_covers = np.zeros(m, dtype=np.intc)
    path = np.zeros((m, 2), dtype=int)
    pi = np.zeros(m, dtype=int)
    munkres_(C, Stars, Primes, row_covers, col_covers, path, pi)
    return pi



#########################
# Backtracing Functions #
#########################

cdef double _BACKTRACE_TOL = 1E-5

def uted_backtrace(x_nodes, x_adj, y_nodes = None, y_adj = None, delta = None):
    """ Computes the unordered tree edit distance between the trees x and y,
    each described by a list of nodes and an adjacency list adj, where adj[i]
    is a list of indices pointing to children of node i. This function
    returns an alignment representation of the distance.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Parameters
    ----------
    x_nodes: list or tuple
        a list of nodes for tree x OR a tuple of the form (x_nodes, x_adj).
    x_adj: list or tuple
        an adjacency list for tree x OR a tuple of the form (y_nodes, y_adj).
    y_nodes: list (default = x_adj[0])
        a list of nodes for tree y.
    y_adj: list (default = x_adj[1])
        an adjacency list for tree y.
    delta: function (default = None)
        a function that takes two nodes as inputs and returns their pairwise
        distance, where delta(x, None) should be the cost of deleting x and
        delta(None, y) should be the cost of inserting y. If undefined, this
        method calls standard_ted instead.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the unordered tree
        edit distance.

    """
    if isinstance(x_nodes, tuple):
        x_nodes, x_adj, y_nodes, y_adj = extract_from_tuple_input(x_nodes, x_adj)

    # the number of nodes in both trees
    cdef int m = len(x_nodes)
    cdef int n = len(y_nodes)

    # if either tree is empty, we can only delete/insert all nodes in the
    # non-empty tree.
    cdef int i = 0
    cdef int j = 0
    ali = Alignment()
    if m == 0:
        for j in range(n):
            ali.append_tuple(-1, j)
        return ali
    if n == 0:
        for i in range(n):
            ali.append_tuple(i, -1)
        return ali

    # Set up an array to store edit costs for replacements,
    # deletions, and insertions
    Delta = np.ones((m+1, n+1))
    cdef double[:,:] Delta_view = Delta

    if delta is None:
        for i in range(m):
            for j in range(n):
                if x_nodes[i] == y_nodes[j]:
                    Delta_view[i, j] = 0.
    else:
        # First, compute all pairwise replacement costs
        for i in range(m):
            for j in range(n):
                Delta_view[i,j] = delta(x_nodes[i], y_nodes[j])
        # Then, compute the deletion and insertion costs
        for i in range(m):
            Delta_view[i,n] = delta(x_nodes[i], None)
        for j in range(n):
            Delta_view[m,j] = delta(None, y_nodes[j])

    # compute the actual tree edit distance
    D_forest, D_tree = _uted(x_nodes, x_adj, y_nodes, y_adj, Delta)


    # x_nodes_print = [str(x) for x in x_nodes] + ['-']
    # y_nodes_print = [str(y) for y in y_nodes] + ['-']
    # print('Delta:')
    # print('i\\j\t\t' + '\t'.join(['%d' % j for j in range(n+1)]))
    # print('\tx_i\\y_j\t' + '\t'.join(y_nodes_print))
    # for i in range(m+1):
      # print(str(i) + '\t' + x_nodes_print[i] + '\t' + '\t'.join(['%g' % d for d in Delta[i, :].tolist()]))

    # print('D_forest:')
    # print('i\\j\t\t' + '\t'.join(['%d' % j for j in range(n+1)]))
    # print('\tx_i\\y_j\t' + '\t'.join(y_nodes_print))
    # for i in range(m+1):
      # print(str(i) + '\t' + x_nodes_print[i] + '\t' + '\t'.join(['%g' % d for d in D_forest[i, :].tolist()]))

    # print('D_tree:')
    # print('i\\j\t\t' + '\t'.join(['%d' % j for j in range(n+1)]))
    # print('\tx_i\\y_j\t' + '\t'.join(y_nodes_print))
    # for i in range(m+1):
      # print(str(i) + '\t' + x_nodes_print[i] + '\t' + '\t'.join(['%g' % d for d in D_tree[i, :].tolist()]))

    # set up temporary variables for backtracing
    cdef int k
    cdef int l
    cdef int i_k
    cdef int j_l
    cdef int m_i
    cdef int n_j
    cdef double rep_cost = 0.
    cdef double del_cost = 0.
    cdef int k_min = 0
    cdef double ins_cost = 0.
    cdef int l_min = 0

    # start backtracing. We perform backtracing via a stack which
    # stores a tuple of indices (i, j) in both trees and a 'mode'
    # flag, namely whether we are currently align the subtrees at
    # (i, j) or the child forests.
    stk = [(0, 0, 0)]
    while stk:
        i, j, mode = stk.pop()
        # print('pop (%d, %d, %d)' % (i, j, mode))
        if mode == 0:
            # --- tree edit mode ---
            # first, check if we only wish to delete/insert
            if i == -1:
                # print('insert %d' % j)
                ali.append_tuple(-1, j)
                for j_l in y_adj[j][::-1]:
                    stk.append((-1, j_l, 0))
                continue
            if j == -1:
                # print('delete %d' % i)
                ali.append_tuple(i, -1)
                for i_k in x_adj[i][::-1]:
                    stk.append((i_k, -1, 0))
                continue
            # print('continuing tree alignment at (%d, %d)' % (i, j))
            m_i = len(x_adj[i])
            n_j = len(y_adj[j])
            # check whether replacement is co-optimal
            # print('D_tree[i, j] = %g; Delta[i, j] = %g, D_forest[i, j] = %g' % (D_tree[i, j], Delta[i, j], D_forest[i, j]))
            if D_tree[i, j] + _BACKTRACE_TOL > Delta[i, j] + D_forest[i, j]:
                # print('replacement co-optimal')
                # if so, i and j are aligned
                ali.append_tuple(i, j)
                # and we need to optimally align all children
                stk.append((i, j, 1))
                continue
            # check whether deletion is co-optimal
            if m_i == 0:
                if D_tree[i, j] + _BACKTRACE_TOL > Delta[i, n] + D_tree[m, j]:
                    # print('deletion co-optimal')
                    # if it is and i has no children,
                    # delete i and insert everything in j
                    ali.append_tuple(i, -1)
                    stk.append((-1, j, 0))
                    continue
            else:
                # we consider all options to delete i and replace
                # it with one of its children
                for k in range(len(x_adj[i])):
                    i_k = x_adj[i][k]
                    del_cost = D_tree[i_k, j] + D_tree[i, n] - D_tree[i_k, n]
                    if D_tree[i, j] + _BACKTRACE_TOL > del_cost:
                        k_min = k
                        break
                if D_tree[i, j] + _BACKTRACE_TOL > del_cost:
                    # print('deletion with child %d co-optimal' % k_min)
                    # delete i
                    ali.append_tuple(i, -1)
                    for k in range(len(x_adj[i])-1, -1, -1):
                        i_k = x_adj[i][k]
                        if k == k_min:
                            # continue the alignment between i_k and j
                            stk.append((i_k, j, 0))
                        else:
                            # delete all children of i except i_k
                            stk.append((i_k, -1, 0))
                    continue
            # check whether insertion is co-optimal
            if n_j == 0:
                if D_tree[i, j] + _BACKTRACE_TOL > Delta[m, j] + D_tree[i, n]:
                    # print('insertion co-optimal')
                    # if it is and j has no children,
                    # insert j and delete everything in i
                    ali.append_tuple(-1, j)
                    stk.append((i, -1, 0))
                    continue
            else:
                # we consider all options to insert j and replace
                # it with one of its children
                for l in range(len(y_adj[j])):
                    j_l = y_adj[j][l]
                    ins_cost = D_tree[i, j_l] + D_tree[m, j] - D_tree[m, j_l]
                    if D_tree[i, j] + _BACKTRACE_TOL > ins_cost:
                        l_min = l
                        break
                if D_tree[i, j] + _BACKTRACE_TOL > ins_cost:
                    # print('insertion with child %d co-optimal' % l_min)
                    # insert j
                    ali.append_tuple(-1, j)
                    for l in range(len(y_adj[j])-1, -1, -1):
                        j_l = y_adj[j][l]
                        if l == l_min:
                            # continue the alignment between i and j_l
                            stk.append((i, j_l, 0))
                        else:
                            # delete all children of j except j_l
                            stk.append((-1, j_l, 0))
                    continue
        elif mode == 1:
            # --- forest edit mode ---
            # print('continuing forest alignment at (%d, %d)' % (i, j))
            m_i = len(x_adj[i])
            n_j = len(y_adj[j])
            # if the children are empty, we need to delete all
            # children of the counterpart
            if m_i == 0:
                for j_l in y_adj[j][::-1]:
                    stk.append((-1, j_l, 0))
                continue
            if n_j == 0:
                for i_k in x_adj[i][::-1]:
                    stk.append((i_k, -1, 0))
                continue
            # next, check if replacement is co-optimal. For this
            # purpose, perform the Munkres/Hungarian algorithm
            # for an optimal alignment

            # prepare a cost matrix for the algorithm
            C = np.zeros((m_i + n_j, m_i + n_j))
            for k in range(m_i):
                i_k = x_adj[i][k]
                for l in range(n_j):
                    j_l = y_adj[j][l]
                    # matching ci with cj means editing the ci'th
                    # child of i to the cj'th child of j.
                    C[k, l] = D_tree[i_k, j_l]
            C[:m_i, n_j:] = INFINITY
            for k in range(m_i):
                # matching c with n_j + c means deleting the
                # c'th child of i
                i_k = x_adj[i][k]
                C[k, n_j + k] = D_tree[i_k, n]
            C[m_i:, :n_j] = INFINITY
            for l in range(n_j):
                # matching m_i + c with c means inserting the
                # c'th child of j
                j_l = y_adj[j][l]
                C[m_i + l, l] = D_tree[m, j_l]
            C[m_i:, n_j:] = 0.
            # call the munkres algorithm
            pi = munkres(C)
            # compute the cost
            rep_cost = 0.
            for k in range(m_i):
                i_k = x_adj[i][k]
                if pi[k] >= n_j:
                    # if pi[k] >= n_j, tree i_k should be deleted
                    rep_cost += D_tree[i_k, n]
                else:
                    # otherwise we replace i_k with j_pi[k]
                    j_l = y_adj[j][pi[k]]
                    rep_cost += D_tree[i_k, j_l]
            for l in range(n_j):
                j_l = y_adj[j][l]
                if pi[m_i + l] < n_j:
                    # if pi[m_i + l] < n_j, tree j_l should be inserted
                    rep_cost += D_tree[m, j_l]
            # check if forest replacement is optimal
            # print('rep_cost = %g with alignment %s' % (rep_cost, str(pi)))
            # print('C matrix was')
            # print('i\\j\t' + '\t'.join(['%d' % j_l for j_l in y_adj[j]]))
            # for k in range(m_i):
              # print(str(x_adj[i][k]) + '\t' + '\t'.join(['%g' % d for d in C[k, :].tolist()]))
            # for l in range(n_j):
              # print(str(y_adj[j][l]) + '\t' + '\t'.join(['%g' % d for d in C[m_i+l, :].tolist()]))
            if D_forest[i, j] + _BACKTRACE_TOL > rep_cost:
                # print('forest replacement is co-optimal')
                # if replacement is co-optimal, perform all
                # deletion/insertion/replacements of subtrees
                # dictated by the Hungarian result
                for l in range(n_j-1, -1, -1):
                    if pi[m_i + l] < n_j:
                        # insert everything in subtree j_l
                        stk.append((-1, y_adj[j][l], 0))
                for k in range(m_i-1, -1, -1):
                    if pi[k] < n_j:
                        # put subtree replacements onto the stack
                        stk.append((x_adj[i][k], y_adj[j][pi[k]], 0))
                    else:
                        # delete everything in subtree i_k
                        stk.append((x_adj[i][k], -1, 0))
                continue
            # if it is not, check if forest deletion is optimal
            for k in range(m_i):
                i_k = x_adj[i][k]
                # accordingly, we need to consider the cost of editing
                # the children of node i_k with the children of j,
                # plus the cost of deleting all other children of i.
                del_cost = D_forest[i_k, j] + D_forest[i, n] - D_forest[i_k, n]
                if D_forest[i, j] + _BACKTRACE_TOL > del_cost:
                    k_min = k
                    break
            if D_forest[i, j] + _BACKTRACE_TOL > del_cost:
                # print('forest deletion with child %d co-optimal' % k_min)
                for k in range(m_i):
                    i_k = x_adj[i][k]
                    if k == k_min:
                        # delete i_k
                        ali.append_tuple(i_k, -1)
                        # continue the forest alignment between
                        # the children of i_k and j
                        stk.append((i_k, j, 1))
                    else:
                        # delete all children of i except i_k
                        stk.append((i_k, -1, 0))
                continue
            # if it is not, check if forest insertion is optimal
            for l in range(n_j):
                j_l = y_adj[j][l]
                # accordingly, we need to consider the cost of editing
                # tree i into tree j_l plus the cost o inserting all other
                # children of j.
                ins_cost = D_forest[i, j_l] + D_forest[m, j] - D_forest[m, j_l]
                if D_forest[i, j] + _BACKTRACE_TOL > ins_cost:
                    l_min = l
                    break
            if D_forest[i, j] + _BACKTRACE_TOL > ins_cost:
                # print('forest insertion with child %d co-optimal' % l_min)
                for l in range(n_j):
                    j_l = y_adj[j][l]
                    if l == l_min:
                        # insert j_l
                        ali.append_tuple(-1, j_l)
                        # continue the forest alignment between
                        # the children of i and j_l
                        stk.append((i, j_l, 1))
                    else:
                        # insert all children of j except j_l
                        stk.append((-1, j_l, 0))
                continue
            raise ValueError('Internal error: no option was co-optimal')
        else:
            raise ValueError('Internal error: Unrecognized mode: %s' % str(mode))
        # if nothing is co-optimal, raise an exception
        raise ValueError('Internal error: No option was co-optimal during alignment of (%d, %d), mode = %s' % (i, j, str(mode)))
    return ali
