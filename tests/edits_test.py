#!/usr/bin/python3
"""
Tests list edits, i.e. functions which take a list as input and
return a changed list.

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
import edist.edits as edits

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2019-2021, Benjamin Paaßen"
__license__ = "GPLv3"
__maintainer__ = "Benjamin Paaßen"
__email__ = "bpaassen@techfak.uni-bielefeld.de"


class TestEdits(unittest.TestCase):

    def test_replacement(self):
        edit = edits.Replacement(1, "a")
        lst = ["b", "b"]
        self.assertEqual(["b", "a"], edit.apply(lst))
        self.assertEqual(["b", "b"], lst)
        edit.apply_in_place(lst)
        self.assertEqual(["b", "a"], lst)
        edit.apply_in_place(lst)
        self.assertEqual(["b", "a"], lst)

    def test_deletion(self):
        edit = edits.Deletion(1)
        lst = ["a", "b", "c"]
        self.assertEqual(["a", "c"], edit.apply(lst))
        self.assertEqual(["a", "b", "c"], lst)
        edit.apply_in_place(lst)
        self.assertEqual(["a", "c"], lst)
        edit.apply_in_place(lst)
        self.assertEqual(["a"], lst)

    def test_insertion(self):
        edit = edits.Insertion(1, "b")
        lst = ["a", "c"]
        self.assertEqual(["a", "b", "c"], edit.apply(lst))
        self.assertEqual(["a", "c"], lst)
        edit.apply_in_place(lst)
        self.assertEqual(["a", "b", "c"], lst)
        edit.apply_in_place(lst)
        self.assertEqual(["a", "b", "b", "c"], lst)

    def test_script(self):
        script1 = edits.Script(
            [edits.Insertion(1, "b"), edits.Deletion(0), edits.Replacement(0, "c")]
        )
        script2 = edits.Script([edits.Replacement(0, "c")])

        lst = ["a", "b"]
        self.assertEqual(["c", "b"], script1.apply(lst))
        self.assertEqual(["c", "b"], script2.apply(lst))
        self.assertEqual(["a", "b"], lst)
        script1.apply_in_place(lst)
        self.assertEqual(["c", "b"], lst)
        script2.apply_in_place(lst)
        self.assertEqual(["c", "b"], lst)

    def test_alignment_to_script(self):
        # consider two lists
        x = ["a", "b", "c"]
        y = ["b", "e", "f", "c"]

        # set up an alignment
        ali = alignment.Alignment()
        ali.append_tuple(0, -1)  # delete a
        ali.append_tuple(1, 0)  # replace b with b
        ali.append_tuple(-1, 1)  # insert e
        ali.append_tuple(-1, 2)  # insert f
        ali.append_tuple(2, 3)  # replace c with c

        # set up the expected script
        expected_script = edits.Script(
            [edits.Deletion(0), edits.Insertion(1, "e"), edits.Insertion(2, "f")]
        )

        # compare with actual script
        actual_script = edits.alignment_to_script(ali, x, y)
        self.assertEqual(expected_script, actual_script)
        self.assertEqual(y, expected_script.apply(x))

        # set up the inverse alignment
        ali = alignment.Alignment()
        ali.append_tuple(-1, 0)  # insert a
        ali.append_tuple(0, 1)  # replace b with b
        ali.append_tuple(1, -1)  # delete e
        ali.append_tuple(2, -1)  # delete f
        ali.append_tuple(3, 2)  # replace c with c

        # set up the expected script
        expected_script = edits.Script(
            [edits.Deletion(2), edits.Deletion(1), edits.Insertion(0, "a")]
        )

        # compare with actual script
        actual_script = edits.alignment_to_script(ali, y, x)
        self.assertEqual(expected_script, actual_script)
        self.assertEqual(x, expected_script.apply(y))


if __name__ == "__main__":
    unittest.main()
