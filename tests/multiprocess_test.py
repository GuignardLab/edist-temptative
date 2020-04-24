#!/usr/bin/python3
"""
Tests parallel computations of tree edit distances.

"""
# Copyright (C) 2019-2020
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

import unittest
import time
import numpy as np
from edist.dtw import dtw
from edist.dtw import dtw_string
from edist.ted import ted
from edist.alignment import Alignment
from edist.sed import standard_sed_backtrace
import edist.multiprocess as multiprocess

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def kron_distance(x, y):
    if(x == y):
        return 0.
    else:
        return 1.

class TestMultiprocess(unittest.TestCase):

    def test_pairwise_dtw(self):
        # consider three example sequences
        Xs = ['abc', 'aabbcc', 'dbc']
        D_expected = np.array([[0, 0, 1], [0, 0, 2], [1, 2, 0]], dtype=float)

        # compute actual distances using dtw
        D_actual = multiprocess.pairwise_distances(Xs, Xs, dist = dtw_string)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using symmetric function
        D_actual = multiprocess.pairwise_distances_symmetric(Xs, dist = dtw_string)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using general dtw and a delta function
        D_actual = multiprocess.pairwise_distances(Xs, Xs, dist = dtw, delta = kron_distance)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using symmetric function
        D_actual = multiprocess.pairwise_distances_symmetric(Xs, dist = dtw, delta = kron_distance)
        np.testing.assert_array_equal(D_expected, D_actual)


    def test_pairwise_ted(self):
        # consider three example trees, one of them being empty
        x = []
        x_adj = []
        # the tree a(b(c, d), e)
        y = ['a', 'b', 'c', 'd', 'e']
        y_adj = [[1, 4], [2, 3], [], [], []]
        # the tree f(g)
        z = ['f', 'g']
        z_adj = [[1], []]

        Xs = [(x, x_adj), (y, y_adj), (z, z_adj)]

        # set up the expected distances
        D_expected = np.array([[0, 5, 2], [5, 0, 5], [2, 5, 0]], dtype=int)

        # compute actual distances using the standard edit distance
        D_actual = multiprocess.pairwise_distances(Xs, Xs, dist = ted)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using symmetric function
        D_actual = multiprocess.pairwise_distances_symmetric(Xs, dist = ted)
        np.testing.assert_array_equal(D_expected, D_actual)


        # compute actual distances using the general edit distance
        D_expected = np.array([[0., 5., 2.], [5., 0., 5.], [2., 5., 0.]])
        D_actual = multiprocess.pairwise_distances(Xs, Xs, dist = ted, delta = kron_distance)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using symmetric function
        D_actual = multiprocess.pairwise_distances_symmetric(Xs, dist = ted, delta = kron_distance)
        np.testing.assert_array_equal(D_expected, D_actual)


    def test_pairwise_backtrace(self):
        # consider three example strings, one of them being empty
        x = ''
        y = 'abcde'
        z = 'fg'

        Xs = [x, y, z]

        # set up expected alignments
        B_expected = [[], [], []]
        ali = Alignment()
        B_expected[0].append(ali)
        ali = Alignment()
        ali.append_tuple(-1, 0)
        ali.append_tuple(-1, 1)
        ali.append_tuple(-1, 2)
        ali.append_tuple(-1, 3)
        ali.append_tuple(-1, 4)
        B_expected[0].append(ali)
        ali = Alignment()
        ali.append_tuple(-1, 0)
        ali.append_tuple(-1, 1)
        B_expected[0].append(ali)
        ali = Alignment()
        ali.append_tuple(0, -1)
        ali.append_tuple(1, -1)
        ali.append_tuple(2, -1)
        ali.append_tuple(3, -1)
        ali.append_tuple(4, -1)
        B_expected[1].append(ali)
        ali = Alignment()
        ali.append_tuple(0, 0)
        ali.append_tuple(1, 1)
        ali.append_tuple(2, 2)
        ali.append_tuple(3, 3)
        ali.append_tuple(4, 4)
        B_expected[1].append(ali)
        ali = Alignment()
        ali.append_tuple(0, 0)
        ali.append_tuple(1, 1)
        ali.append_tuple(2, -1)
        ali.append_tuple(3, -1)
        ali.append_tuple(4, -1)
        B_expected[1].append(ali)
        ali = Alignment()
        ali.append_tuple(0, -1)
        ali.append_tuple(1, -1)
        B_expected[2].append(ali)
        ali = Alignment()
        ali.append_tuple(0, 0)
        ali.append_tuple(1, 1)
        ali.append_tuple(-1, 2)
        ali.append_tuple(-1, 3)
        ali.append_tuple(-1, 4)
        B_expected[2].append(ali)
        ali = Alignment()
        ali.append_tuple(0, 0)
        ali.append_tuple(1, 1)
        B_expected[2].append(ali)

        # compute actual backtraces using the standard edit distance
        B_actual = multiprocess.pairwise_backtraces(Xs, Xs, dist_backtrace = standard_sed_backtrace)
        self.assertEqual(B_expected, B_actual)

if __name__ == '__main__':
    unittest.main()
