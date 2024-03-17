#!python
#cython: language_level=3
"""
Implements algebraic dynamic programming for edit distances.
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

import heapq
import random
from collections.abc import Callable
import numpy as np
from edist.alignment import Alignment

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.2'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class RuleEntry:
    """ Models the set of all grammar rules for a single nonterminal symbol
    in an ADP grammar.

    Attributes
    ----------
    _reps: list
        A list of possible replacements, stored as tuples of the form
       (rep, B), where rep is the name of a replacement operation and
       B is the nonterminal we obtain after applying rep.
    _dels: list
        A list of possible deletions, stored as tuples of the form
       (del, B), where del is the name of a deletion operation and
       B is the nonterminal we obtain after applying del.
    _inss: list
        A list of possible insertions, stored as tuples of the form
       (ins, B), where ins is the name of a insertion operation and
       B is the nonterminal we obtain after applying ins.

    """
    def __init__(self):
        self._reps = []
        self._dels = []
        self._inss = []

    def __repr__(self):
        op_str = 'replacements: '
        for rep_edge in self._reps:
            op_str += 'via ' + rep_edge[0] + ' to ' + rep_edge[1] + ' '
        op_str += '\ndeletions: '
        for del_edge in self._dels:
            op_str += 'via ' + del_edge[0] + ' to ' + del_edge[1] + ' '
        op_str += '\ninsertions: '
        for ins_edge in self._inss:
            op_str += 'via ' + ins_edge[0] + ' to ' + ins_edge[1] + ' '
        return op_str

    def __str__(self):
        return self.__repr__()


def string_to_index_map(lst):
    """ Inverts a list of objects, i.e. converts a list of objects to
    a map from objects to indices in the list.

    Parameters
    ----------
    lst: list
        A list of objects [`x_1`, ..., `x_m`].

    Returns
    -------
    dct: dictionary
        A map where dct[x_i] = i for all i.

    """
    dct = {}
    for i in range(len(lst)):
        dct[lst[i]] = i
    return dct

def string_to_index_list(lst, dct):
    """ Converts a list of objects to an index list, given a index mapping.

    Parameters
    ----------
    lst: list
        A list of objects [`x_1`, ..., `x_m`].
    dct: dictionary
        A mapping from objects to indices.

    Returns
    -------
    idx_list: list
        The list [dct[`x_1`], ..., dct[`x_m`]]

    """
    idx_list = []
    for e in lst:
        if(e not in dct):
            raise ValueError('unknown entry: %s' % str(e))
        idx_list.append(dct[e])
    return idx_list

def string_to_index_tuple_list(lst, op_dct, nont_dct):
    """ Converts a list of tuples to a list of tuple-indices.

    Parameters
    ----------
    lst: list
        A list of tuples [(`x_1`, `y_1`), ..., (`x_m`, `y_m`)].
    op_dct: dictionary
        A mapping from x-objects to indices.
    nont_dct: dictionary
        A mapping from y-objects to indices.

    Returns
    -------
    idx_list: list
        The list [(op_dct[`x_1`], nont_dct[`y_1`]), ...,
        (op_dct[`x_m`], nont_dct[`y_m`])]

    """
    idx_list = []
    for e in lst:
        op_idx = op_dct[e[0]]
        if(op_idx is None):
            raise ValueError('unknown operation string: %s' % str(e[0]))
        nont_idx = nont_dct[e[1]]
        if(nont_idx is None):
            raise ValueError('unknown nonterminal string: %s' % str(e[1]))
        idx_list.append((op_idx, nont_idx))
    return idx_list

class Grammar:
    """ Models an ADP grammar, consisting of a starting nonterminal, a set of
    accepting nonterminals, a set of permitted edit operations and a set of
    rules of the form

    `A` -> `delta` `B`

    where `A` and `B` are nonterminals and `delta` is an edit operation, either
    a replacement, a deletion, or an insertion. We define the set of possible
    edit scripts permitted by a given grammar inductively as all scripts that
    can be produced by starting from the self._start, replacing the current
    nonterminal with the right-hand-side of a matching rule arbitrary many
    times, and then deleting the current nonterminal if it is accepting.

    Attributes
    ----------
    _nonterminals: list (default = _accepting + {start})
        A list of nonterminals of this grammar. Defaults to the union of
        accepting and start.
    _start: str
        The starting nonterminal, which should be in _nonterminals.
    _accepting: list
        A list of accepting nonterminals, all of which should be in
        self._nonterminals.
    _reps: list (default = [])
        A list of names of possible replacement operations.
        Defaults to an empty list.
    _dels: list (default = [])
        A list of names of possible deletion operations.
        Defaults to an empty list.
    _inss: list (default = [])
        A list of names of possible insertion operations.
        Defaults to an empty list.
    _rules: dictionary (default = {})
        A mapping of nonterminal symbols to RuleEntries. Refer to
        The documentation above for more details on RuleEntries.
        Defaults to an empty map.

    """
    def __init__(self, start, accepting, nonterminals = None, reps = None, dels = None, inss = None, rules = None):
        self._start = start
        self._accepting = accepting
        if(nonterminals is None):
            self._nonterminals = []
        else:
            self._nonterminals = nonterminals
        if(reps is None):
            self._reps = []
        else:
            self._reps = reps
        if(dels is None):
            self._dels = []
        else:
            self._dels = dels
        if(inss is None):
            self._inss = []
        else:
            self._inss = inss
        if(rules is None):
            self._rules = {}
        else:
            self._rules = rules
        self._initialize_rule_entry(start)
        for nont in accepting:
            self._initialize_rule_entry(nont)

    def __repr__(self):
        op_str = 'Start at ' + str(self._start) + '. Rules:\n'
        for nont in self._nonterminals:
            op_str += 'From ' + nont 
            if(nont in self._accepting):
                op_str += ' (accepting)'
            op_str += ':\n'
            op_str += str(self._rules[nont])
        return op_str

    def __str__(self):
        return self.__repr__()

    def _initialize_rule_entry(self, source):
        if(source not in self._nonterminals):
            self._nonterminals.append(source)
        if(source not in self._rules):
            self._rules[source] = RuleEntry()

    def append_replacement(self, source, target, operation):
        """ Appends a rule to this grammar for a replacement operation,
        i.e. a rule of the form `A` -> `rep` `B`.

        Parameters
        ----------
        source: str
            The left-hand-side nonterminal `A` of the new rule.
            If not in self._nonterminals yet, it is appended automatically.
        target: str
            The right-hand-side nonterminal `B` of the new rule.
            If not in self._nonterminals yet, it is appended automatically.
        operation: str
            The name of the replacement operation rep. If not in
            self._reps yet, it is appended automatically.

        """
        if(operation not in self._reps):
            self._reps.append(operation)
        self._initialize_rule_entry(source)
        self._initialize_rule_entry(target)
        self._rules[source]._reps.append((operation, target))

    def append_deletion(self, source, target, operation):
        """ Appends a rule to this grammar for a deletion operation,
        i.e. a rule of the form `A` -> `del` `B`.

        Parameters
        ----------
        source: str
            The left-hand-side nonterminal `A` of the new rule.
            If not in self._nonterminals yet, it is appended automatically.
        target: str
            The right-hand-side nonterminal `B` of the new rule.
            If not in self._nonterminals yet, it is appended automatically.
        operation: str
            The name of the deletion operation del. If not in
            self._dels yet, it is appended automatically.
        """
        if(operation not in self._dels):
            self._dels.append(operation)
        self._initialize_rule_entry(source)
        self._initialize_rule_entry(target)
        self._rules[source]._dels.append((operation, target))

    def append_insertion(self, source, target, operation):
        """ Appends a rule to this grammar for a deletion operation,
        i.e. a rule of the form `A` -> `ins` `B`.

        Parameters
        ----------
        source: str
            The left-hand-side nonterminal `A` of the new rule.
            If not in self._nonterminals yet, it is appended automatically.
        target: str
            The right-hand-side nonterminal `B` of the new rule.
            If not in self._nonterminals yet, it is appended automatically.
        operation: str
            The name of the insertion operation ins. If not in
            self._inss yet, it is appended automatically.

        """
        if(operation not in self._inss):
            self._inss.append(operation)
        self._initialize_rule_entry(source)
        self._initialize_rule_entry(target)
        self._rules[source]._inss.append((operation, target))

    def size(self):
        """ Returns the number of nonterminals in this grammar.

        Returns
        -------
        size: int
            the number of nonterminals in this grammar.

        """
        return len(self._nonterminals)

    def start(self):
        """ Returns the starting nonterminal of this grammar.

        Returns
        -------
        start: str
            the starting nonterminal of this grammar.

        """
        return self._start

    def nonterminals(self):
        """ Returns the list of nonterminals of this grammar.

        Returns
        -------
        nonts: list
            the list of nonterminals of this grammar.

        """
        return self._nonterminals

    def validate(self, deltas):
        """ Ensures that this grammar is compatible with the given algebra
        deltas.

        Parameters
        ----------
        deltas: dictionary
            An algebra, i.e. a mapping from operation names to distance
            functions.

        Raises
        ------
        ValueError
            If any of the operations of this grammar is not supported by the
            given algebra.

        """
        for rep_op in self._reps:
            if(rep_op not in deltas):
                raise ValueError('costs for the ' + rep_op + ' operation are undefined!')
        for del_op in self._dels:
            if(del_op not in deltas):
                raise ValueError('costs for the ' + del_op + ' operation are undefined!')
        for ins_op in self._inss:
            if(ins_op not in deltas):
                raise ValueError('costs for the ' + ins_op + ' operation are undefined!')

    def adjacency_lists(self):
        """ Returns the adjacency list format of this grammar.

        Returns
        -------
        start_idx: int
            The index of the starting nonterminal symbol.
        accpt_idxs: list
            A list of indices for all accepting nonterminals.
        rep_adj: list
            An adjacency list covering all replacement operations,
            i.e. a list where the `i`-th entry contains all rules of the form
            `A` -> `rep` `B`, where `A` is the `i`-th nonterminal and the
            right-hand-sides are represented as tuples of operation indices and
            nonterminal indices.
        del_adj: list
            An adjacency list covering all deletion operations,
            i.e. a list where the `i`-th entry contains all rules of the form
            `A` -> `del` `B`, where `A` is the `i`-th nonterminal and the
            right-hand-sides are represented as tuples of operation indices and
            nonterminal indices.
        ins_adj: list
            An adjacency list covering all insertion operations,
            i.e. a list where the `i`-th entry contains all rules of the form
            `A` -> `ins` `B`, where `A` is the `i`-th nonterminal and the
            right-hand-sides are represented as tuples of operation indices and
            nonterminal indices.

        """
        # first, create maps from string to index representations
        nont_map = string_to_index_map(self._nonterminals)
        reps_map = string_to_index_map(self._reps)
        dels_map = string_to_index_map(self._dels)
        inss_map = string_to_index_map(self._inss)
        # translate the start symbol to an index
        start_idx = nont_map[self._start]
        # translate the accepting symbol list
        accpt_idxs = string_to_index_list(self._accepting, nont_map)
        # then, create an adjacency list representation for all replacements,
        # deletions, and insertions separately
        rep_adj = []
        del_adj = []
        ins_adj = []
        for nont in self._nonterminals:
            rule_entry = self._rules[nont]
            rep_adj.append(string_to_index_tuple_list(rule_entry._reps, reps_map, nont_map))
            del_adj.append(string_to_index_tuple_list(rule_entry._dels, dels_map, nont_map))
            ins_adj.append(string_to_index_tuple_list(rule_entry._inss, inss_map, nont_map))
        return start_idx, accpt_idxs, rep_adj, del_adj, ins_adj

    def inverse_adjacency_lists(self):
        """ Returns the inverse adjacency list format of this grammar.

        Returns
        -------
        start_idx: int
            The index of the starting nonterminal symbol.
        accpt_idxs: list
            A list of indices for all accepting nonterminals.
        rep_adj: list
            An adjacency list covering all replacement operations,
            i.e. a list where the `i`-th entry contains all rules of the form
            `A` -> `rep` `B`, where `B` is the `i`-th nonterminal and
            (`A`, `rep`) is represented as a tuple tuples of operation index
            and nonterminal index.
        del_adj: list
            An adjacency list covering all deletion operations,
            i.e. a list where the `i`-th entry contains all rules of the form
            `A` -> `del` `B`, where `B` is the `i`-th nonterminal and
            (`A`, `del`) is represented as a tuple tuples of operation index
            and nonterminal index.
        ins_adj: list
            An adjacency list covering all insertion operations,
            i.e. a list where the `i`-th entry contains all rules of the form
            `A` -> `ins` `B`, where `B` is the `i`-th nonterminal and
            (`A`, `ins`) is represented as a tuple tuples of operation index
            and nonterminal index.

        """
        # first, create maps from string to index representations
        nont_map = string_to_index_map(self._nonterminals)
        reps_map = string_to_index_map(self._reps)
        dels_map = string_to_index_map(self._dels)
        inss_map = string_to_index_map(self._inss)
        # translate the start symbol to an index
        start_idx = nont_map[self._start]
        # translate the accepting symbol list
        accpt_idxs = string_to_index_list(self._accepting, nont_map)
        # then, create an adjacency list representation for all replacements,
        # deletions, and insertions separately
        rep_adj = []
        del_adj = []
        ins_adj = []
        for nont in self._nonterminals:
            rep_adj.append([])
            del_adj.append([])
            ins_adj.append([])
        for A in self._nonterminals:
            rule_entry = self._rules[A]
            for (delta, B) in rule_entry._reps:
                rep_adj[nont_map[B]].append((reps_map[delta], nont_map[A]))
            for (delta, B) in rule_entry._dels:
                del_adj[nont_map[B]].append((dels_map[delta], nont_map[A]))
            for (delta, B) in rule_entry._inss:
                ins_adj[nont_map[B]].append((inss_map[delta], nont_map[A]))
        return start_idx, accpt_idxs, rep_adj, del_adj, ins_adj

def edit_distance(x, y, grammar, deltas):
    """ Computes the edit distance between two sequences x and y, based on
    the given ADP grammar and the given algebra.

    Parameters
    ----------
    x: list
        A list of objects.
    y: list
        Another list of objects.
    grammar: class adp.Grammar
        An ADP grammar. Refer to the documentation above for more information.
    deltas: dictionary
        An algebra, i.e. a mapping from operation names to distance functions
        OR a single distance function if the grammar supports only a single
        replacement, deletion, and insertion operation.

    Returns
    -------
    d: float
        The edit distance between x and y.

    """
    # apply the internal edit distance function
    Ds, _, _, _, start_idx, _, _, _, _ = _edit_distance(x, y, grammar, deltas)
    return Ds[start_idx, 0, 0]

def _edit_distance(x, y, grammar, deltas):
    """ Computes the edit distance including all internal variables
    necessary during computation.

    Parameters
    ----------
    x: list
        A list of objects of length m.
    y: list
        Another list of objects of length n.
    grammar: class adp.Grammar
        An ADP grammar with R nonterminals, K_rep replacements,
        K_del deletions, and K_ins insertions. Refer to the documentation
        above for more information.
    deltas: dictionary
        An algebra, i.e. a mapping from operation names to distance functions
        OR a single distance function if the grammar supports only a single
        replacement, deletion, and insertion operation.

    Returns
    -------
    Ds: array_like
        A R x m+1 x n+1 tensor containing the dynamic programming matrices
        for all nonterminals.
    Deltas_rep: array_like
        A K_rep x m x n tensor containing the replacement costs for
        all replacements.
    Deltas_del: array_like
        A K_rep x m matrix containing the deletion costs for
        all deletions.
    Deltas_ins: array_like
        A K_ins x n matrix containing the insertion costs for
        all insertions.
    start_idx: int
        The index of the starting nonterminal.
    accpt_idxs: list
        The indices of accepting nonterminals.
    adj_rep: list
        An adjacency list representation of the grammar rules for
        replacement operations.
    adj_del: list
        An adjacency list representation of the grammar rules for
        deletion operations.
    adj_ins: list
        An adjacency list representation of the grammar rules for
        insertion operations.

    """
    # check if the given algebra is compatible with the given grammar.
    if(isinstance(deltas, Callable)):
        if(len(grammar._reps) > 1 or len(grammar._dels) > 1 or len(grammar._inss) > 1):
            raise ValueError('If a function is given instead of an algebra, the grammar can only support a single operation of each type; otherwise, ambiguities arise.')
        # generate a mock algebra that is definitely compatible
        delta = deltas
        deltas = {grammar._reps[0] : delta, grammar._dels[0] : delta, grammar._inss[0] : delta}
    else:
        grammar.validate(deltas)

    cdef int m = len(x)
    cdef int n = len(y)
    # pre-compute all operation costs

    # First, compute all pairwise replacements
    cdef int K_rep = len(grammar._reps)
    Deltas_rep = np.zeros((K_rep, m, n))
    cdef double[:,:,:] Deltas_rep_view = Deltas_rep
    cdef int i
    cdef int j
    cdef int k
    for k in range(K_rep):
        delta = deltas[grammar._reps[k]]
        for i in range(m):
            for j in range(n):
                Deltas_rep_view[k, i, j] = delta(x[i], y[j])

    # Then, compute all deletions
    cdef int K_del = len(grammar._dels)
    Deltas_del = np.zeros((K_del, m))
    cdef double[:,:] Deltas_del_view = Deltas_del
    for k in range(K_del):
        delta = deltas[grammar._dels[k]]
        for i in range(m):
            Deltas_del_view[k, i] = delta(x[i], None)

    # Then, compute all insertions
    cdef int K_ins = len(grammar._inss)
    Deltas_ins = np.zeros((K_ins, n))
    cdef double[:,:] Deltas_ins_view = Deltas_ins
    for k in range(K_ins):
        delta = deltas[grammar._inss[k]]
        for j in range(n):
            Deltas_ins_view[k, j] = delta(None, y[j])

    # retrieve the adjacency list representation for
    # the grammar
    start_idx, accpt_idxs, adj_rep, adj_del, adj_ins = grammar.adjacency_lists()

    # Initialize the dynamic programming matrices
    # for all nonterminals
    cdef int R = len(grammar._nonterminals)
    Ds = np.full((R, m+1, n+1), np.inf)
    cdef double[:,:,:] Ds_view = Ds
    # initialize last entry for all accepting symbols
    cdef int nont
    for nont in accpt_idxs:
        Ds_view[nont, m, n] = 0.

    # initialize last column for all symbols
    cdef int r
    cdef int s
    for i in range(m-1,-1,-1):
        for r in range(R):
            for (k, s) in adj_del[r]:
                Ds_view[r, i, n] = Deltas_del_view[k, i] + Ds_view[s, i+1, n]

    # initialize last row for all symbols
    for j in range(n-1,-1,-1):
        for r in range(R):
            for (k, s) in adj_ins[r]:
                Ds_view[r, m, j] = Deltas_ins_view[k, j] + Ds_view[s, m, j+1]

    # perform the remaining computation
    cdef double min_cost
    cdef double current_cost
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            for r in range(R):
                min_cost = np.inf
                # first, consider replacements
                for (k, s) in adj_rep[r]:
                    current_cost = Deltas_rep_view[k, i, j] + Ds_view[s, i+1, j+1]
                    if(current_cost < min_cost):
                        min_cost = current_cost
                # then, consider deletions
                for (k, s) in adj_del[r]:
                    current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
                    if(current_cost < min_cost):
                        min_cost = current_cost
                # finally, consider insertions
                for (k, s) in adj_ins[r]:
                    current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
                    if(current_cost < min_cost):
                        min_cost = current_cost
                # set new entry to minimum
                Ds_view[r, i, j] = min_cost

    return Ds, Deltas_rep, Deltas_del, Deltas_ins, start_idx, accpt_idxs, adj_rep, adj_del, adj_ins


####### BACKTRACING FUNCTIONS #######

cdef double _BACKTRACE_TOL = 1E-5

def backtrace(x, y, grammar, deltas):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the given ADP grammar and algebra. This mechanism
    is deterministic and will always prefer replacements over other options.

    Parameters
    ----------
    x: list
        A list of objects.
    y: list
        Another list of objects.
    grammar: class adp.Grammar
        An ADP grammar. Refer to the documentation above for more information.
    deltas: dictionary
        An algebra, i.e. a mapping from operation names to distance functions
        OR a single distance function if the grammar supports only a single
        replacement, deletion, and insertion operation.

    Returns
    -------
    alignment: class alignment.Alignment
        a co-optimal alignment between x and y.

    """
    # apply the internal edit distance function
    Ds, Deltas_rep, Deltas_del, Deltas_ins, start_idx, accpt_idxs, adj_rep, adj_del, adj_ins = _edit_distance(x, y, grammar, deltas)

    # declare c variables
    cdef int m = len(x)
    cdef int n = len(y)
    cdef int i
    cdef int j
    cdef int k
    cdef int r
    cdef int s

    cdef double[:,:,:] Ds_view = Ds
    cdef double[:,:, :] Deltas_rep_view = Deltas_rep
    cdef double[:,:] Deltas_del_view = Deltas_del
    cdef double[:,:] Deltas_ins_view = Deltas_ins

    # then, perform the backtracing
    r = start_idx
    i = 0
    j = 0
    cdef int found_coopt
    alignment = Alignment()
    while(i < m and j < n):
        # check which alignment option is co-optimal
        found_coopt = False
        # check all possible replacements
        for (k, s) in adj_rep[r]:
            current_cost = Deltas_rep_view[k, i, j] + Ds_view[s, i+1, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # replacement is co-optimal
                alignment.append_tuple(i, j, grammar._reps[k])
                r = s
                i += 1
                j += 1
                found_coopt = True
                break
        if(found_coopt):
            continue
        # check all possible deletions
        for (k, s) in adj_del[r]:
            current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # deletion is co-optimal
                alignment.append_tuple(i, -1, grammar._dels[k])
                r = s
                i += 1
                found_coopt = True
                break
        if(found_coopt):
            continue
        # check all possible insertions
        for (k, s) in adj_ins[r]:
            current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                alignment.append_tuple(-1, j, grammar._inss[k])
                r = s
                j += 1
                found_coopt = True
                break
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')
    # delete all demaining nodes
    while(i < m):
        found_coopt = False
        # check all possible deletions
        for (k, s) in adj_del[r]:
            current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # deletion is co-optimal
                alignment.append_tuple(i, -1, grammar._dels[k])
                r = s
                i += 1
                found_coopt = True
                break
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')
    # insert all demaining nodes
    while(j < n):
        found_coopt = False
        # check all possible deletions
        for (k, s) in adj_ins[r]:
            current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                alignment.append_tuple(-1, j, grammar._inss[k])
                r = s
                j += 1
                found_coopt = True
                break
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')
    if(r not in accpt_idxs):
        raise ValueError('Ended in a non-accepting nonterminal')
    return alignment

def backtrace_stochastic(x, y, grammar, deltas):
    """ Computes a co-optimal alignment between the two input sequences
    x and y, given the given ADP grammar and algebra. This mechanism
    is stochastic and will return a random alignment.

    Note that the randomness does _not_ produce a uniform distribution over
    all co-optimal alignments because random choices at the start of the
    alignment process dominate. If you wish to characterize the overall
    distribution accurately, use backtrace_matrix instead.

    Parameters
    ----------
    x: list
        A list of objects.
    y: list
        Another list of objects.
    grammar: class adp.Grammar
        An ADP grammar. Refer to the documentation above for more information.
    deltas: dictionary
        An algebra, i.e. a mapping from operation names to distance functions
        OR a single distance function if the grammar supports only a single
        replacement, deletion, and insertion operation.

    Returns
    -------
    alignment: class alignment.Alignment
        a co-optimal alignment between x and y.

    """
    # apply the internal edit distance function
    Ds, Deltas_rep, Deltas_del, Deltas_ins, start_idx, accpt_idxs, adj_rep, adj_del, adj_ins = _edit_distance(x, y, grammar, deltas)

    # declare c variables
    cdef int m = len(x)
    cdef int n = len(y)
    cdef int i
    cdef int j
    cdef int k
    cdef int r
    cdef int s
    cdef int c
    cdef int op

    cdef double[:,:,:] Ds_view = Ds
    cdef double[:,:,:] Deltas_rep_view = Deltas_rep
    cdef double[:,:] Deltas_del_view = Deltas_del
    cdef double[:,:] Deltas_ins_view = Deltas_ins

    # then, perform the backtracing
    r = start_idx
    i = 0
    j = 0
    cdef int found_coopt
    alignment = Alignment()
    while(i < m and j < n):
        # check which alignment option is co-optimal
        # check all possible replacements
        coopts = []
        for (k, s) in adj_rep[r]:
            current_cost = Deltas_rep_view[k, i, j] + Ds_view[s, i+1, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # replacement is co-optimal
                coopts.append((0, k, s))
        # check all possible deletions
        for (k, s) in adj_del[r]:
            current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # deletion is co-optimal
                coopts.append((1, k, s))
        # check all possible insertions
        for (k, s) in adj_ins[r]:
            current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                coopts.append((2, k, s))
        if(not coopts):
            raise ValueError('Internal error: No option is co-optimal.')
        # select a random operation
        c = random.randrange(len(coopts))
        # apply the corresponding operation
        op = coopts[c][0]
        k  = coopts[c][1]
        r  = coopts[c][2]
        if(op == 0):
            # apply replacement
            alignment.append_tuple(i, j, grammar._reps[k])
            i += 1
            j += 1
        elif(op == 1):
            # apply deletion
            alignment.append_tuple(i, -1, grammar._dels[k])
            i += 1
        elif(op == 2):
            # apply insertion
            alignment.append_tuple(-1, j, grammar._inss[k])
            j += 1
    # delete all demaining nodes
    while(i < m):
        coopts = []
        # check all possible deletions
        for (k, s) in adj_del[r]:
            current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # deletion is co-optimal
                coopts.append((k, s))
        if(not coopts):
            raise ValueError('Internal error: No option is co-optimal.')
        # select a random deletion
        c = random.randrange(len(coopts))
        # apply the corresponding operation
        k  = coopts[c][0]
        r  = coopts[c][1]
        alignment.append_tuple(i, -1, grammar._dels[k])
        i += 1

    # insert all demaining nodes
    while(j < n):
        coopts = []
        # check all possible insertions
        for (k, s) in adj_ins[r]:
            current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                coopts.append((k, s))
        if(not coopts):
            raise ValueError('Internal error: No option is co-optimal.')
        # select a random insertion
        c = random.randrange(len(coopts))
        # apply the corresponding operation
        k  = coopts[c][0]
        r  = coopts[c][1]
        alignment.append_tuple(-1, j, grammar._inss[k])
        j += 1
    if(r not in accpt_idxs):
        raise ValueError('Ended in a non-accepting nonterminal')
    return alignment


def backtrace_matrix(x, y, grammar, deltas):
    """ Computes three tensors, P_rep, P_del, and P_ins, which summarize all
    co-optimal alignments between x and y.

    In particular, P_rep[k, i, j] specifies the fraction of co-optimal
    alignments in which node x[i] has been replaced with node y[j] using
    operation grammar._reps[k]. Accordingly, P_del[k, i] specifies the
    fraction of co-optimal alignments in which x[i] has been deleted using
    operation grammar._dels[k], and P_ins[k, j] specifies the fraction of
    co-optimal alignments in which y[j] has been inserted using operation
    grammar._inss[k].

    Parameters
    ----------
    x: list
        A list of objects.
    y: list
        Another list of objects.
    grammar: class adp.Grammar
        An ADP grammar. Refer to the documentation above for more information.
    deltas: dictionary
        An algebra, i.e. a mapping from operation names to distance functions
        OR a single distance function if the grammar supports only a single
        replacement, deletion, and insertion operation.

    Returns
    -------
    P_rep: array_like
        a tensor where P_rep[k, i, j] specifies the fraction of co-optimal
        alignments in which node x[i] has been replaced with node y[j] using
        operation grammar._reps[k].
    P_del: array_like
        a matrix where P_del[k, i] specifies the fraction of co-optimal
        alignments in which x[i] has been deleted using operation
        grammar._dels[k].
    P_ins: array_like
        a matrix where P_ins[k, j] specifies the fraction of co-optimal
        alignments in which y[j] has been inserted using operation
        grammar._inss[k].
    k: int
        the number of co-optimal alignments.

    """
    # apply the internal edit distance function
    Ds, Deltas_rep, Deltas_del, Deltas_ins, start_idx, accpt_idxs, adj_rep, adj_del, adj_ins = _edit_distance(x, y, grammar, deltas)

    # declare c variables
    cdef int m = len(x)
    cdef int n = len(y)
    cdef int i
    cdef int j
    cdef int k
    cdef int r
    cdef int s
    cdef int c
    cdef int op

    cdef double[:,:,:] Ds_view = Ds
    cdef double[:,:,:] Deltas_rep_view = Deltas_rep
    cdef double[:,:] Deltas_del_view = Deltas_del
    cdef double[:,:] Deltas_ins_view = Deltas_ins

    # compute the forward tensor Alpha, which contains the number of
    # co-optimal alignment paths from cell [start_idx, 0, 0] to cell [r, i, j]
    Alpha = np.zeros((len(grammar._nonterminals), m+1, n+1), dtype=int)
    cdef long[:,:, :] Alpha_view = Alpha
    Alpha_view[start_idx, 0, 0] = 1
    # build a queue of cells which we still need to process
    q = [(start_idx, 0, 0)]
    # build a set which stores the already visited cells
    visited = set()
    # initialize temporary variables
    cdef int found_coopt = False
    cdef long num_coopts = 0
    while(q):
        (r, i, j) = heapq.heappop(q)
        if((i, j) in visited):
            continue
        visited.add((r, i, j))
        num_coopts = Alpha_view[r, i, j]
        found_coopt = False
        if(i == m):
            if(j == n):
                continue
            # if we are at the end of the first sequence, we can only insert
            # check all possible insertions
            for (k, s) in adj_ins[r]:
                current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
                if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                    # insertion is co-optimal
                    Alpha_view[s, i, j+1] += num_coopts
                    heapq.heappush(q, (s, i, j+1))
                    found_coopt = True
            if(not found_coopt):
                raise ValueError('Internal error: No option is co-optimal.')
            continue
        if(j == n):
            # if we are at the end of the second sequence, we can only delete
            # check all possible deletions
            for (k, s) in adj_del[r]:
                current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
                if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                    # deletion is co-optimal
                    Alpha_view[s, i+1, j] += num_coopts
                    heapq.heappush(q, (s, i+1, j))
                    found_coopt = True
            if(not found_coopt):
                raise ValueError('Internal error: No option is co-optimal.')
            continue
        # check which alignment option is co-optimal
        # check all possible replacements
        for (k, s) in adj_rep[r]:
            current_cost = Deltas_rep_view[k, i, j] + Ds_view[s, i+1, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # replacement is co-optimal
                Alpha_view[s, i+1, j+1] += num_coopts
                heapq.heappush(q, (s, i+1, j+1))
                found_coopt = True
        # check all possible deletions
        for (k, s) in adj_del[r]:
            current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # deletion is co-optimal
                Alpha_view[s, i+1, j] += num_coopts
                heapq.heappush(q, (s, i+1, j))
                found_coopt = True
        # check all possible insertions
        for (k, s) in adj_ins[r]:
            current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                Alpha_view[s, i, j+1] += num_coopts
                heapq.heappush(q, (s, i, j+1))
                found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    # compute the backward matrix Beta, which contains the number of
    # co-optimal alignment paths from cell [s, i, j] to cells [accepting, m, n]
    Beta = np.zeros((len(grammar._nonterminals), m+1, n+1), dtype=int)
    cdef long[:,:,:] Beta_view = Beta
    for r in accpt_idxs:
        Beta_view[r, m, n] = 1
    # retrieve the inverse adjacency list format of the grammar
    _, _, inv_adj_rep, inv_adj_del, inv_adj_ins = grammar.inverse_adjacency_lists()

    # iterate in downward lexigraphic order over the visited cells
    for (s, i, j) in sorted(visited, key = (lambda tpl : (tpl[1], tpl[2], tpl[0])), reverse = True):
        num_coopts = Beta_view[s, i, j]
        found_coopt = False
        if(i == 0):
            if(j == 0):
                continue
            # if we are at the start of the first sequence, we can only insert
            # check all possible insertions
            for (k, r) in inv_adj_ins[s]:
                current_cost = Deltas_ins_view[k, j-1] + Ds_view[s, i, j]
                if(Ds_view[r, i, j-1] + _BACKTRACE_TOL > current_cost):
                    # insertion is co-optimal
                    Beta_view[r, i, j-1] += num_coopts
                    found_coopt = True
            if(not found_coopt):
                raise ValueError('Internal error: No option is co-optimal.')
            continue
        if(j == 0):
            # if we are at the start of the second sequence, we can only delete
            # check all possible deletions
            for (k, r) in inv_adj_del[s]:
                current_cost = Deltas_del_view[k, i-1] + Ds_view[s, i, j]
                if(Ds_view[r, i-1, j] + _BACKTRACE_TOL > current_cost):
                    # insertion is co-optimal
                    Beta_view[r, i-1, j] += num_coopts
                    found_coopt = True
            if(not found_coopt):
                raise ValueError('Internal error: No option is co-optimal.')
            continue
        # check which alignment option is co-optimal
        # check all possible replacements
        for (k, r) in inv_adj_rep[s]:
            current_cost = Deltas_rep_view[k, i-1, j-1] + Ds_view[s, i, j]
            if(Ds_view[r, i-1, j-1] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                Beta_view[r, i-1, j-1] += num_coopts
                found_coopt = True
        # check all possible deletions
        for (k, r) in inv_adj_del[s]:
            current_cost = Deltas_del_view[k, i-1] + Ds_view[s, i, j]
            if(Ds_view[r, i-1, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                Beta_view[r, i-1, j] += num_coopts
                found_coopt = True
        # check all possible insertions
        for (k, r) in inv_adj_ins[s]:
            current_cost = Deltas_ins_view[k, j-1] + Ds_view[s, i, j]
            if(Ds_view[r, i, j-1] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                Beta_view[r, i, j-1] += num_coopts
                found_coopt = True
        if(not found_coopt):
            raise ValueError('Internal error: No option is co-optimal.')

    if(np.sum(Alpha[:, m, n]) != Beta_view[start_idx, 0, 0]):
        raise ValueError('Internal error: Alignment count in Alpha and Beta matrix did not agree; got %d versus %d' % (np.sum(Alpha[:, m, n]), Beta_view[start_idx, 0, 0]))

    # compute the output matrices, specifying how often each operation was used
    K_rep = np.zeros((len(grammar._reps), m, n), dtype=int)
    cdef long[:,:,:] K_rep_view = K_rep
    K_del = np.zeros((len(grammar._dels), m), dtype=int)
    cdef long[:,:] K_del_view = K_del
    K_ins = np.zeros((len(grammar._inss), n), dtype=int)
    cdef long[:,:] K_ins_view = K_ins
    for (r, i, j) in visited:
        num_coopts = Alpha_view[r, i, j]
        if(i == m):
            if(j == n):
                continue
            # if we are at the end of the first sequence, we can only insert
            # check all possible insertions
            for (k, s) in adj_ins[r]:
                current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
                if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                    # insertion is co-optimal
                    K_ins_view[k, j] += num_coopts * Beta_view[s, i, j+1]
            continue
        if(j == n):
            # if we are at the end of the second sequence, we can only delete
            # check all possible deletions
            for (k, s) in adj_del[r]:
                current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
                if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                    # deletion is co-optimal
                    K_del_view[k, i] += num_coopts * Beta_view[s, i+1, j]
            continue
        # check which alignment option is co-optimal
        # check all possible replacements
        for (k, s) in adj_rep[r]:
            current_cost = Deltas_rep_view[k, i, j] + Ds_view[s, i+1, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # replacement is co-optimal
                K_rep_view[k, i, j] += num_coopts * Beta_view[s, i+1, j+1]
        # check all possible deletions
        for (k, s) in adj_del[r]:
            current_cost = Deltas_del_view[k, i] + Ds_view[s, i+1, j]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # deletion is co-optimal
                K_del_view[k, i] += num_coopts * Beta_view[s, i+1, j]
        # check all possible insertions
        for (k, s) in adj_ins[r]:
            current_cost = Deltas_ins_view[k, j] + Ds_view[s, i, j+1]
            if(Ds_view[r, i, j] + _BACKTRACE_TOL > current_cost):
                # insertion is co-optimal
                K_ins_view[k, j] += num_coopts * Beta_view[s, i, j+1]

    # compute the final summary matrices by dividing K by the overall number
    # of co-optimal alignments
    num_coopts = Beta_view[start_idx, 0, 0]
    P_rep = K_rep.astype(float) / num_coopts
    P_del = K_del.astype(float) / num_coopts
    P_ins = K_ins.astype(float) / num_coopts
    return P_rep, P_del, P_ins, num_coopts
