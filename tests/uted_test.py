#!/usr/bin/python3
"""
Tests the unordered tree edit distance implementation.

"""
# Copyright (C) 2019-2021
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
import edist.uted as uted
import numpy as np
from edist.alignment import Alignment
from edist import tree_utils

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2019-2021, Benjamin Paaßen"
__license__ = "GPLv3"
__maintainer__ = "Benjamin Paaßen"
__email__ = "benjamin.paassen@hu-berlin.de"


class TestUTED(unittest.TestCase):

    def test_uted(self):

        # test a trivial example: aligning a single leaf
        x_nodes = ["a"]
        x_adj = [[]]
        y_nodes = ["a", "c", "d", "e", "f"]
        y_adj = [[1, 4], [2, 3], [], [], []]

        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj)
        self.assertAlmostEqual(4.0, d)

        # test symmetry
        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj)
        self.assertAlmostEqual(4.0, d)

        # test equivalence with hand-defined unit cost
        def delta(x, y):
            if x == y:
                return 0.0
            else:
                return 1.0

        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj, delta)
        self.assertAlmostEqual(4.0, d)

        # test symmetry
        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj, delta)
        self.assertAlmostEqual(4.0, d)

        # test an example with a single free node
        x_nodes = ["a", "e"]
        x_adj = [[1], []]
        y_nodes = ["a", "c", "d", "e", "f"]
        y_adj = [[1, 4], [2, 3], [], [], []]

        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj)
        self.assertAlmostEqual(3.0, d)

        # test symmetry
        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj)
        self.assertAlmostEqual(3.0, d)

        # test an example with two full trees
        x_nodes = ["a", "b", "c", "e", "d"]
        x_adj = [[1], [2], [3, 4], [], []]
        y_nodes = ["a", "c", "d", "e", "f"]
        y_adj = [[1, 4], [2, 3], [], [], []]

        d = uted.uted(x_nodes, x_adj, y_nodes, y_adj)
        self.assertAlmostEqual(2.0, d)

        # test symmetry
        d = uted.uted(y_nodes, y_adj, x_nodes, x_adj)
        self.assertAlmostEqual(2.0, d)

    def test_uted_backtrace(self):
        # test a trivial example: aligning a single leaf
        x_nodes = ["a"]
        x_adj = [[]]
        y_nodes = ["a", "c", "d", "e", "f"]
        y_adj = [[1, 4], [2, 3], [], [], []]
        alignment = uted.uted_backtrace(x_nodes, x_adj, y_nodes, y_adj)
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(-1, 1)
        expected_alignment.append_tuple(-1, 2)
        expected_alignment.append_tuple(-1, 3)
        expected_alignment.append_tuple(-1, 4)
        self.assertEqual(expected_alignment, alignment)

        # test an example with two full trees
        x_nodes = ["a", "b", "c", "e", "d"]
        x_adj = [[1], [2], [3, 4], [], []]
        y_nodes = ["a", "c", "d", "e", "f"]
        y_adj = [[1, 4], [2, 3], [], [], []]
        alignment = uted.uted_backtrace(x_nodes, x_adj, y_nodes, y_adj)
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, -1)
        expected_alignment.append_tuple(2, 1)
        expected_alignment.append_tuple(3, 3)
        expected_alignment.append_tuple(4, 2)
        expected_alignment.append_tuple(-1, 4)
        self.assertEqual(expected_alignment, alignment)

    def test_uted_backtrace_big(self):
        # Test case devised by Marcel Wittmund (RWTH).
        x_nodes = [14, 13, 15, 20, 21, 94, 83, 74, 70, 84, 28, 29, 37, 23]
        x_adj = [
            [1, 2, 3, 13],
            [],
            [],
            [4, 10],
            [5],
            [6],
            [7, 9],
            [8],
            [],
            [],
            [11, 12],
            [],
            [],
            [],
        ]
        y_nodes = [
            29,
            28,
            30,
            31,
            39,
            40,
            16,
            52,
            53,
            55,
            53,
            66,
            67,
            68,
            73,
            35,
            36,
            112,
            103,
            92,
            84,
            93,
            147,
            159,
            104,
            50,
            51,
            58,
            38,
        ]
        y_adj = [
            [1, 2, 15, 28],
            [],
            [3, 4, 5, 6, 7, 8, 9],
            [],
            [],
            [],
            [],
            [],
            [],
            [10, 11, 12, 13, 14],
            [],
            [],
            [],
            [],
            [],
            [16, 25],
            [17],
            [18],
            [19, 24],
            [20, 21],
            [],
            [22, 23],
            [],
            [],
            [],
            [26, 27],
            [],
            [],
            [],
        ]

        self.assertEqual(len(x_nodes), len(x_adj))
        self.assertEqual(len(y_nodes), len(y_adj))

        tree_utils.check_dfs_structure(x_adj)
        tree_utils.check_dfs_structure(y_adj)

        def delta3(x, y):
            if x is None or y is None:
                return 1.0
            if x == y:
                return 0.0
            else:
                return 1.0

        alignment = uted.uted_backtrace(x_nodes, x_adj, y_nodes, y_adj)
        d_align = alignment.cost(x_nodes, y_nodes, delta3)
        d_uted = uted.uted(x_nodes, x_adj, y_nodes, y_adj, delta3)
        self.assertEqual(d_align, d_uted)

    def test_munkres(self):
        C = np.array([[7.0, 5.0, 11.2], [5.0, 4.0, 1.0], [9.3, 3.0, 2.0]])
        pi = uted.munkres(C)
        np.testing.assert_array_equal([0, 2, 1], pi)

        x = ["a", "b", "c", "d"]
        y = ["b", "e", "d"]

        C = np.full((len(x) + len(y), len(x) + len(y)), np.inf)
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] == y[j]:
                    C[i, j] = 0.0
                else:
                    C[i, j] = 1.0
        for i in range(len(x)):
            C[i, len(y) + i] = 1.0
        for j in range(len(y)):
            C[len(x) + j, j] = 1.0
        C[len(x) :, len(y) :] = 0.0
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
        self.assertTrue(np.all(np.abs(x - y[pi]) < 1.0))

        # test another matrix which lead to a bug before
        C = np.array(
            [
                [1.0, 2.0, 1.0, np.inf, np.inf],
                [0.0, 2.0, np.inf, 2.0, np.inf],
                [0.0, 2.0, np.inf, np.inf, 2.0],
                [2.0, np.inf, 0.0, 0.0, 0.0],
                [np.inf, 3.0, 0.0, 0.0, 0.0],
            ]
        )

        pi = uted.munkres(C)
        np.testing.assert_array_equal([2, 1, 0, 4, 3], pi)


if __name__ == "__main__":
    unittest.main()
