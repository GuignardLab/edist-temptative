#!/usr/bin/python3
"""
Tests parallel computations of tree edit distances.

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
import unittest
import time
import numpy as np
import edist.multiprocess as multiprocess
import edist.ted as ted

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def kron_distance(x, y):
    if(x == y):
        return 0.
    else:
        return 1.

class TestMultiprocess(unittest.TestCase):

    def test_pairwise_distances(self):
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
        D_actual = multiprocess.pairwise_distances(Xs, Xs, dist = ted.ted)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using symmetric function
        D_actual = multiprocess.pairwise_distances_symmetric(Xs, dist = ted.ted)
        np.testing.assert_array_equal(D_expected, D_actual)


        # compute actual distances using the general edit distance
        D_expected = np.array([[0., 5., 2.], [5., 0., 5.], [2., 5., 0.]])
        D_actual = multiprocess.pairwise_distances(Xs, Xs, dist = ted.ted, delta = kron_distance)
        np.testing.assert_array_equal(D_expected, D_actual)

        # compute again using symmetric function
        D_actual = multiprocess.pairwise_distances_symmetric(Xs, dist = ted.ted, delta = kron_distance)
        np.testing.assert_array_equal(D_expected, D_actual)

if __name__ == '__main__':
    unittest.main()
