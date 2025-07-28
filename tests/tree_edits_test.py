#!/usr/bin/python3
"""
Tests tree edits, i.e. functions which take a tree as input and
return a changed tree.

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
import edist.alignment as alignment
import edist.tree_edits as tree_edits

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2019-2021, Benjamin Paaßen"
__license__ = "GPLv3"
__maintainer__ = "Benjamin Paaßen"
__email__ = "bpaassen@techfak.uni-bielefeld.de"


class TestTreeEdits(unittest.TestCase):

    def test_replacement(self):
        # generate a simple tree
        nodes = ["a", "b", "c", "d", "e"]
        adj = [[1, 4], [2, 3], [], [], []]
        # generate a replacement
        rep1 = tree_edits.Replacement(0, "f")
        # set up expected result
        expected_nodes = ["f", "b", "c", "d", "e"]
        expected_adj = adj
        # apply edit
        actual_nodes, actual_adj = rep1.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)
        # apply another edit
        rep2 = tree_edits.Replacement(1, "g")
        expected_nodes = ["a", "g", "c", "d", "e"]
        expected_adj = adj
        actual_nodes, actual_adj = rep2.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)

        # test again with in-place edits
        expected_nodes = ["f", "g", "c", "d", "e"]
        expected_adj = adj
        rep1.apply_in_place(nodes, adj)
        rep2.apply_in_place(nodes, adj)
        self.assertEqual(expected_nodes, nodes)
        self.assertEqual(expected_adj, adj)

    def test_deletion(self):
        # generate a simple tree
        nodes = ["a", "b", "c", "d", "e"]
        adj = [[1, 4], [2, 3], [], [], []]
        # generate a deletion
        del1 = tree_edits.Deletion(4)
        # set up expected result
        expected_nodes = ["a", "b", "c", "d"]
        expected_adj = [[1], [2, 3], [], []]
        # apply edit
        actual_nodes, actual_adj = del1.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)
        # apply another deletion
        del2 = tree_edits.Deletion(1)
        expected_nodes = ["a", "c", "d", "e"]
        expected_adj = [[1, 2, 3], [], [], []]
        actual_nodes, actual_adj = del2.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)

        # test deleting the root node
        del3 = tree_edits.Deletion(0)
        expected_nodes = ["b", "c", "d", "e"]
        expected_adj = [[1, 2], [], [], []]
        actual_nodes, actual_adj = del3.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)

        # test again with in-place edits
        expected_nodes = ["a", "c", "d"]
        expected_adj = [[1, 2], [], []]
        del1.apply_in_place(nodes, adj)
        del2.apply_in_place(nodes, adj)
        self.assertEqual(expected_nodes, nodes)
        self.assertEqual(expected_adj, adj)

    def test_insertion(self):
        # generate a simple tree
        nodes = ["f", "g"]
        adj = [[1], []]
        # insert a d as new child of f
        ins1 = tree_edits.Insertion(0, 1, "d")
        # set up expected result
        expected_nodes = ["f", "g", "d"]
        expected_adj = [[1, 2], [], []]
        # apply edit
        actual_nodes, actual_adj = ins1.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)
        # insert an e as new root node
        ins2 = tree_edits.Insertion(-1, 1, "e")
        expected_nodes = ["f", "g", "e"]
        expected_adj = [[1], [], []]
        actual_nodes, actual_adj = ins2.apply(nodes, adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)
        # insert an a as new parent of f and e
        ins3 = tree_edits.Insertion(-1, 0, "a", 2)
        expected_nodes = ["a", "f", "g", "e"]
        expected_adj = [[1, 3], [2], [], []]
        actual_nodes, actual_adj = ins3.apply(actual_nodes, actual_adj)
        # check result
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)

        # test again with in-place edits
        expected_nodes = ["a", "f", "g", "d", "e"]
        expected_adj = [[1, 4], [2, 3], [], [], []]
        ins1.apply_in_place(nodes, adj)
        ins2.apply_in_place(nodes, adj)
        ins3.apply_in_place(nodes, adj)
        self.assertEqual(expected_nodes, nodes)
        self.assertEqual(expected_adj, adj)

    def test_script(self):
        # consider two trees
        x_nodes = ["f", "g"]
        x_adj = [[1], []]
        y_nodes = ["a", "b", "c", "d", "e"]
        y_adj = [[1, 4], [2, 3], [], [], []]
        # construct a script that maps from x to y
        xy_script = tree_edits.Script()
        xy_script.append(tree_edits.Insertion(-1, 0, "a", 1))
        xy_script.append(tree_edits.Replacement(1, "b"))
        xy_script.append(tree_edits.Replacement(2, "c"))
        xy_script.append(tree_edits.Insertion(1, 1, "d"))
        xy_script.append(tree_edits.Insertion(0, 1, "e"))
        # apply script
        actual_y_nodes, actual_y_adj = xy_script.apply(x_nodes, x_adj)
        # check result
        self.assertEqual(y_nodes, actual_y_nodes)
        self.assertEqual(y_adj, actual_y_adj)
        # construct a script that maps from y to x
        yx_script = tree_edits.Script()
        yx_script.append(tree_edits.Deletion(4))
        yx_script.append(tree_edits.Deletion(3))
        yx_script.append(tree_edits.Replacement(2, "g"))
        yx_script.append(tree_edits.Replacement(1, "f"))
        yx_script.append(tree_edits.Deletion(0))
        # apply script
        actual_x_nodes, actual_x_adj = yx_script.apply(y_nodes, y_adj)
        # check result
        self.assertEqual(x_nodes, actual_x_nodes)
        self.assertEqual(x_adj, actual_x_adj)

    def test_get_num_descendants(self):
        adj = [[1, 4], [2, 3], [], [], []]
        filter_set = [0, 3, 4]
        expected_num_descendants = [1, 1, 0, 0, 0]
        actual_num_descendants = tree_edits.num_descendants(adj, filter_set)
        self.assertEqual(expected_num_descendants, actual_num_descendants)

    def test_alignment_to_script(self):
        # consider two trees
        x_nodes = ["f", "g"]
        x_adj = [[1], []]
        y_nodes = ["a", "b", "c", "d", "e"]
        y_adj = [[1, 4], [2, 3], [], [], []]
        # and a alignment mapping between them
        xy_alignment = alignment.Alignment()
        xy_alignment.append_tuple(-1, 0)  # insert a
        xy_alignment.append_tuple(0, 1)  # replace f with b
        xy_alignment.append_tuple(1, 2)  # replace g with c
        xy_alignment.append_tuple(-1, 3)  # insert d
        xy_alignment.append_tuple(-1, 4)  # insert e
        # set up expected script
        xy_script = tree_edits.Script()
        xy_script.append(tree_edits.Replacement(0, "b"))
        xy_script.append(tree_edits.Replacement(1, "c"))
        xy_script.append(tree_edits.Insertion(-1, 0, "a", 1))
        xy_script.append(tree_edits.Insertion(1, 1, "d"))
        xy_script.append(tree_edits.Insertion(0, 1, "e"))
        # convert to script
        actual_xy_script = tree_edits.alignment_to_script(
            xy_alignment, x_nodes, x_adj, y_nodes, y_adj
        )
        # check result
        self.assertEqual(xy_script, actual_xy_script)

        # construct an inverse alignment
        yx_alignment = alignment.Alignment()
        yx_alignment.append_tuple(4, -1)  # delete e
        yx_alignment.append_tuple(3, -1)  # delete d
        yx_alignment.append_tuple(2, 1)  # replace c with g
        yx_alignment.append_tuple(1, 0)  # replace b with f
        yx_alignment.append_tuple(0, -1)  # delete a
        # set up expected script
        yx_script = tree_edits.Script()
        yx_script.append(tree_edits.Replacement(2, "g"))
        yx_script.append(tree_edits.Replacement(1, "f"))
        yx_script.append(tree_edits.Deletion(4))
        yx_script.append(tree_edits.Deletion(3))
        yx_script.append(tree_edits.Deletion(0))
        # convert to script
        actual_yx_script = tree_edits.alignment_to_script(
            yx_alignment, y_nodes, y_adj, x_nodes, x_adj
        )
        # check result
        self.assertEqual(yx_script, actual_yx_script)


if __name__ == "__main__":
    unittest.main()
