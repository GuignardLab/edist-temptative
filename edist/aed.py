"""
Implements a sequence edit distance with affine gap costs using ADP.

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

import numpy as np
import edist.adp as adp

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

# first, define the grammar
_grammar = adp.Grammar('A', ['A', 'D', 'I'])
_grammar.append_replacement('A', 'A', 'rep')
_grammar.append_deletion('A', 'D', 'del')
_grammar.append_insertion('A', 'I', 'ins')
_grammar.append_replacement('D', 'A', 'rep')
_grammar.append_deletion('D', 'D', 'skdel')
_grammar.append_insertion('D', 'I', 'ins')
_grammar.append_replacement('I', 'A', 'rep')
_grammar.append_insertion('I', 'I', 'skins')

class AffineAlgebra:
    """ This is a class to efficiently store an algebra for the affine edit
    distance grammar in a pickleable format.

    Attributes
    ----------
    _rep: function (default = Kronecker distance)
        A function for replacement costs, i.e. _rep(x, y) is the cost of
        replacing x with y.
    _gap: function (default = constant function with 1.0)
        A function for deletion/insertion costs, i.e. _gap(x) is the cost of
        deleting/inserting x.
    _gap_cost: float (default = 1.0)
        a constant cost for deletions/insertions.
    _skip: function (default = constant function with 0.5)
        A function for deletion/insertion extension costs, i.e. _skip(x) is the
        cost of skip-deleting/-inserting x.
    _skip_cost: float (default = 0.5)
        a constant cost for deletion/insertion extensions.

    """
    def __init__(self, rep = None, gap = 1., skip = 0.5):
        if(rep is None):
            self._rep = self._kron
        else:
            self._rep = rep
        if(not callable(gap)):
            self._gap = self._gap_const
            self._gap_cost = gap
        else:
            self._gap = gap
        if(not callable(skip)):
            self._skip = self._skip_const
            self._skip_cost = skip
        else:
            self._skip = skip

    def __getitem__(self, key):
        if(key == 'rep'):
            return self._rep
        if(key == 'del' or key == 'ins'):
            return self._gap
        if(key == 'skdel' or key == 'skins'):
            return self._skip
        raise ValueError('Unsupported key: %s' % str(key))

    def __contains__(self, item):
        return item in ['rep', 'del', 'ins', 'skdel', 'skins']

    def _kron(self, x, y):
        if(x == y):
            return 0.
        else:
            return 1.

    def _gap_const(self, x, y):
        return self._gap_cost

    def _skip_const(self, x, y):
        return self._skip_cost

def aed(x, y, rep = None, gap = 1., skip = 0.5):
    """ Computes the affine edit distance using algebraic dynamic programming.

    Parameters
    ----------
    x: list
        A list-like object.
    y: list
        Another list-like object.
    rep: function (default = Kronecker delta)
        A function with two arguments, computing the cost for replacing the
        first with the second OR an AffineAlgebra object, in which case the
        remaining aguments will be ignored. Defaults to the Kronecker distance.
    gap: function or float (default = 1.0)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second OR a number defining a constant cost.
        Defaults to 1.
    skip: function or float (default = 0.5)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second for gap extensions OR a number defining
        a constant cost. Defaults to 0.5.

    Returns
    -------
    d: float
        The affine edit distance between x and y.

    """
    if(isinstance(rep, AffineAlgebra)):
        algebra = rep
    else:
        algebra = AffineAlgebra(rep, gap, skip)
    return adp.edit_distance(x, y, _grammar, algebra)

def aed_backtrace(x, y, rep = None, gap = 1., skip = 0.5):
    """ Computes the backtrace of the affine edit distance using algebraic
    dynamic programming.

    Parameters
    ----------
    x: list
        A list-like object.
    y: list
        Another list-like object.
    rep: function (default = Kronecker delta)
        A function with two arguments, computing the cost for replacing the
        first with the second OR an AffineAlgebra object, in which case the
        remaining aguments will be ignored. Defaults to the Kronecker distance.
    gap: function or float (default = 1.0)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second OR a number defining a constant cost.
        Defaults to 1.
    skip: function or float (default = 0.5)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second for gap extensions OR a number defining
        a constant cost. Defaults to 0.5.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the affine edit
        distance.

    """
    if(isinstance(rep, AffineAlgebra)):
        algebra = rep
    else:
        algebra = AffineAlgebra(rep, gap, skip)
    return adp.backtrace(x, y, _grammar, algebra)

def aed_backtrace_stochastic(x, y, rep = None, gap = 1., skip = 0.5):
    """ Computes the backtrace of the affine edit distance using algebraic
    dynamic programming stochastically.

    Note that the randomness does _not_ produce a uniform distribution over
    all co-optimal alignments because random choices at the start of the
    alignment process dominate. If you wish to characterize the overall
    distribution accurately, use aed_backtrace_matrix instead.

    Parameters
    ----------
    x: list
        A list-like object.
    y: list
        Another list-like object.
    rep: function (default = Kronecker delta)
        A function with two arguments, computing the cost for replacing the
        first with the second OR an AffineAlgebra object, in which case the
        remaining aguments will be ignored. Defaults to the Kronecker distance.
    gap: function or float (default = 1.0)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second OR a number defining a constant cost.
        Defaults to 1.
    skip: function or float (default = 0.5)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second for gap extensions OR a number defining
        a constant cost. Defaults to 0.5.

    Returns
    -------
    alignment: class alignment.Alignment
        A co-optimal alignment between x and y according to the affine edit
        distance.

    """
    if(isinstance(rep, AffineAlgebra)):
        algebra = rep
    else:
        algebra = AffineAlgebra(rep, gap, skip)
    return adp.backtrace_stochastic(x, y, _grammar, algebra)

def aed_backtrace_matrix(x, y, rep = None, gap = 1., skip = 0.5):
    """ Computes the backtrace matrix P of the affine edit distance using
    algebraic dynamic programming.

    In particular, P[i, j] contains the probability of node i being replaced
    with node j in a co-optimal alignment. The last two columns contain
    deletion and deletion-extension probabilities, the last two rows contains
    insertion and insertion-extension probabilities.

    Parameters
    ----------
    x: list
        A list-like object.
    y: list
        Another list-like object.
    rep: function (default = Kronecker distance)
        A function with two arguments, computing the cost for replacing the
        first with the second OR an AffineAlgebra object, in which case the
        remaining aguments will be ignored. Defaults to the Kronecker distance.
    gap: function or float (default = 1.0)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second OR a number defining a constant cost.
        Defaults to 1.
    skip: function or float (default = 0.5)
        A function with two arguments, computing the cost for deleting the
        first or inserting the second for gap extensions OR a number defining
        a constant cost. Defaults to 0.5.

    Returns
    -------
    P: array_like
        A len(x) + 2 x len(y) + 2 matrix where P[i, j] contains the probability
        of node i being replaced with node j in a co-optimal alignment. The last
        two columns contain deletion and deletion-extension probabilities, the
        last two rows contains insertion and insertion-extension probabilities.
    k: int
        The number of co-optimal alignments.

    """
    if(isinstance(rep, AffineAlgebra)):
        algebra = rep
    else:
        algebra = AffineAlgebra(rep, gap, skip)
    P_rep, P_del, P_ins, k = adp.backtrace_matrix(x, y, _grammar, algebra)
    # re-format the matrices
    P = np.zeros((len(x) + 2, len(y) + 2))
    P[:len(x), :][:, :len(y)] = P_rep[0, :, :]
    P[:len(x), len(y):] = P_del.T
    P[len(x):, :len(y)] = P_ins
    # return the result
    return P, k
