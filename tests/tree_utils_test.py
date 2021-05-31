#!/usr/bin/python3
"""
Tests general utility functions to process trees.

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

import unittest
import os
import edist.tree_utils as tree_utils

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestTreeUtils(unittest.TestCase):

    def assertNoException(self, fun, args):
        got_exception = False
        try:
            fun(args)
        except Exception:
            got_exception = True
        self.assertFalse(got_exception)

    def assertException(self, fun, args):
        got_exception = False
        try:
            fun(args)
        except Exception:
            got_exception = True
        self.assertTrue(got_exception)

    def test_root(self):
        self.assertEqual(0, tree_utils.root([[]]))
        self.assertEqual(1, tree_utils.root([[], [0]]))
        self.assertEqual(2, tree_utils.root([[], [], [0,1]]))

    def test_check_tree_structure(self):
        # the empty tree should be valid
        self.assertNoException(tree_utils.check_tree_structure, [])
        # test multiple valid trees
        self.assertNoException(tree_utils.check_tree_structure, [[1], []])
        self.assertNoException(tree_utils.check_tree_structure, [[1, 4], [2, 3], [], [], []])
        self.assertNoException(tree_utils.check_tree_structure, [[1, 2], [3, 4], [], [], []])
        # test an invalid tree, i.e. a DAG with multiple parents
        self.assertException(tree_utils.check_tree_structure, [[1, 3], [2], [], [2]])

    def test_check_dfs_structure(self):
        # the empty tree should be valid
        self.assertNoException(tree_utils.check_dfs_structure, [])
        # test multiple valid trees
        self.assertNoException(tree_utils.check_dfs_structure, [[1], []])
        self.assertNoException(tree_utils.check_dfs_structure, [[1, 4], [2, 3], [], [], []])
        # test a valid tree which is not in DFS order
        self.assertException(tree_utils.check_dfs_structure, [[1, 2], [3, 4], [], [], []])

    def test_to_dfs_structure(self):
        # set up a tree _not_ in dfs structure
        nodes = ['C', 'B', 'D', 'A', 'E']
        adj   = [[], [0, 2], [], [1, 4], []]
        # re-order it
        nodes_actual, adj_actual = tree_utils.to_dfs_structure(nodes, adj)
        # check the result
        nodes_expected = ['A', 'B', 'C', 'D', 'E']
        adj_expected   = [[1, 4], [2, 3], [], [], []]
        self.assertEqual(nodes_expected, nodes_actual)
        self.assertEqual(adj_expected, adj_actual)

    def test_json_read_and_write(self):
        # construct a simple tree
        nodes = ['A', 'B', 'C', 'D', 'E']
        adj   = [[1, 4], [2, 3], [], [], []]
        # write it to a JSON file
        tree_utils.to_json('test_json_tree_file.json', nodes, adj)
        # load it from the JSON file again
        nodes_read, adj_read = tree_utils.from_json('test_json_tree_file.json')
        # check the result
        self.assertEqual(nodes, nodes_read)
        self.assertEqual(adj, adj_read)
        # delete the temporary file
        os.remove('test_json_tree_file.json')

    def test_tree_to_string(self):
        # set up a tree
        nodes = ['C', 'B', 'D', 'A', 'E']
        adj   = [[], [0, 2], [], [1, 4], []]
        # translate it to a string
        expected_string = 'A(B(C, D), E)'
        actual_string = tree_utils.tree_to_string(nodes, adj)
        # check the result
        self.assertEqual(expected_string, actual_string)

    def test_subtree(self):
        # set up a tree
        nodes = ['C', 'B', 'D', 'A', 'E']
        adj   = [[], [0, 2], [], [1, 4], []]
        # extract the subtree rooted at B
        expected_nodes = ['B', 'C', 'D']
        expected_adj = [[1, 2], [], []]
        actual_nodes, actual_adj = tree_utils.subtree(nodes, adj, 1)
        # check the result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)

if __name__ == '__main__':
    unittest.main()
