#!/usr/bin/python3
"""
Tests the set edit distance implementation.

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
import numpy as np
from edist.alignment import Alignment
import edist.seted as seted

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2019-2021, Benjamin Paaßen"
__license__ = "GPLv3"
__maintainer__ = "Benjamin Paaßen"
__email__ = "bpaassen@techfak.uni-bielefeld.de"


class TestSetED(unittest.TestCase):

    def test_seted(self):
        x = []
        y = []
        expected = 0.0
        actual = seted.seted(x, y)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )

        x = ["a", "b", "c"]
        y = []
        expected = 3.0
        actual = seted.seted(x, y)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )
        actual = seted.seted(y, x)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )

        y = ["c", "d", "d", "b"]
        expected = 2.0
        actual = seted.seted(x, y)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )
        actual = seted.seted(y, x)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )

        def custom_delta(x, y):
            if x == y:
                return 0.0
            elif (x == "a" and y is not None) or (y == "a" and x is not None):
                return 5.0
            else:
                return 1.0

        expected = 3.0
        actual = seted.seted(x, y, delta=custom_delta)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )
        actual = seted.seted(y, x, delta=custom_delta)
        self.assertTrue(
            np.abs(expected - actual) < 1e-3,
            "Expected %g but got %g" % (expected, actual),
        )

    def test_sed_backtrace(self):
        x = ["a", "b", "c"]
        y = []

        expected_ali = Alignment()
        expected_ali.append_tuple(0, -1)
        expected_ali.append_tuple(1, -1)
        expected_ali.append_tuple(2, -1)

        actual_ali = seted.seted_backtrace(x, y)
        self.assertEqual(expected_ali, actual_ali)

        expected_ali = Alignment()
        expected_ali.append_tuple(-1, 0)
        expected_ali.append_tuple(-1, 1)
        expected_ali.append_tuple(-1, 2)

        actual_ali = seted.seted_backtrace(y, x)
        self.assertEqual(expected_ali, actual_ali)

        y = ["c", "d", "b"]

        expected_ali = Alignment()
        expected_ali.append_tuple(0, 1)
        expected_ali.append_tuple(1, 2)
        expected_ali.append_tuple(2, 0)

        actual_ali = seted.seted_backtrace(x, y)
        self.assertEqual(expected_ali, actual_ali)

        def custom_delta(x, y):
            if x == y:
                return 0.0
            elif (x == "a" and y is not None) or (y == "a" and x is not None):
                return 5.0
            else:
                return 1.0

        expected_ali = Alignment()
        expected_ali.append_tuple(0, -1)
        expected_ali.append_tuple(1, 2)
        expected_ali.append_tuple(2, 0)
        expected_ali.append_tuple(-1, 1)

        actual_ali = seted.seted_backtrace(x, y, custom_delta)
        self.assertEqual(expected_ali, actual_ali)


if __name__ == "__main__":
    unittest.main()
