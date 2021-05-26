#!/usr/bin/python3
"""
Tests the unordered tree edit distance implementation.

"""
# Copyright (C) 2021
# Benjamin Paaßen
# Humboldt-University of Berlin

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
import uted
import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@hu-berlin.de'

class TestUTED(unittest.TestCase):

#    def test_uted_constrained(self):
#        # test a trivial example: aligning a single leaf
#        x_nodes = ['a']
#        x_adj   = [[]]
#        y_nodes = ['a', 'c', 'd', 'e', 'f']
#        y_adj   = [[1, 4], [2, 3], [], [], []]

#        def delta(x, y):
#            if x == y:
#                return 0.
#            else:
#                return 1.

#        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj, delta)
#        self.assertAlmostEqual(4., d)

#        # test symmetry
#        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj, delta)
#        self.assertAlmostEqual(4., d)

#        # test an example with a single free node
#        x_nodes = ['a', 'e']
#        x_adj   = [[1], []]
#        y_nodes = ['a', 'c', 'd', 'e', 'f']
#        y_adj   = [[1, 4], [2, 3], [], [], []]

#        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj, delta)
#        self.assertAlmostEqual(3., d)

#        # test symmetry
#        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj, delta)
#        self.assertAlmostEqual(3., d)


#        # test an example with two full trees
#        x_nodes = ['a', 'b', 'c', 'e', 'd']
#        x_adj   = [[1], [2], [3, 4], [], []]
#        y_nodes = ['a', 'c', 'd', 'e', 'f']
#        y_adj   = [[1, 4], [2, 3], [], [], []]

#        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj, delta)
#        self.assertAlmostEqual(2., d)

#        # test symmetry
#        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj, delta)
#        self.assertAlmostEqual(2., d)

    def test_munkres(self):
        C = np.array([[7., 5., 11.2], [5., 4., 1.], [9.3, 3., 2.]])
        pi = uted.munkres(C)
        np.testing.assert_array_equal([0, 2, 1], pi)

        x = ['a', 'b', 'c', 'd']
        y = ['b', 'e', 'd']

        C = np.full((len(x) + len(y), len(x) + len(y)), np.inf)
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] == y[j]:
                    C[i, j] = 0.
                else:
                    C[i, j] = 1.
        for i in range(len(x)):
            C[i, len(y)+i] = 1.
        for j in range(len(y)):
            C[len(x)+j, j] = 1.
        C[len(x):, len(y):] = 0.
        pi = uted.munkres(C)
        np.testing.assert_array_equal([3, 0, 1, 2], pi[:4])

        m = 10
        x = np.arange(m)
        y = x + np.random.randn(m) * 0.1
        y = np.random.permutation(y)

        C = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                C[i, j] = abs(x[i] - y[j])

        pi = uted.munkres(C)
        self.assertTrue(np.all(np.abs(x - y[pi]) < 1.))

if __name__ == '__main__':
    unittest.main()
