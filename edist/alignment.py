"""
Implements an alignment between two sequences or trees.

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

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class Tuple:
    """ Models a single alignment entry with an edit operation name,
    a left index, and a right index.
    """
    def __init__(self, name, left, right):
        self._name = name
        self._left = left
        self._right = right

    def cost(self, x, y, deltas):
        """ Computes the cost of the current edit tuple.

        Args:
        x: A symbol list for the left indices.
        y: A symbol list for the right indices.
        deltas: A map which contains for any tuple name
                a function which assigns costs to symbol pairings.
                If provided, the cost for any oepration is rendered as
                well.

        Returns: The cost assigned by deltas to this tuple.
        """
        if(self._left >= 0):
            left = x[self._left]
        else:
            left = None
        if(self._right >= 0):
            right = y[self._right]
        else:
            right = None
        if(self._name):
            delta = deltas[self._name]
            return delta(left, right)
        else:
            return deltas(left, right)


    def render(self, x, y, deltas = None):
        """ Represents an tuple as a string, showing the left
        and right indices in addition to the respective labels in x and y,
        and in addition to the tuple cost.

        Args:
        x: A symbol list for the left indices.
        y: A symbol list for the right indices.
        deltas: (optional) A map which contains for any tuple name
                a function which assigns costs to symbol pairings.
                If provided, the cost for any oepration is rendered as
                well.

        Returns: A string representing this tuple.
        """
        op_str = ''
        if(self._name):
            op_str += str(self._name)
            op_str += ': '
        if(self._left >= 0):
            left = x[self._left]
            op_str += str(left) + ' [%d]' % self._left
        else:
            left = None
            op_str += '-'
        op_str += ' vs. '
        if(self._right >= 0):
            right = y[self._right]
            op_str += str(right) + ' [%d]' % self._right
        else:
            right = None
            op_str += '-'
        if(deltas):
            op_str += ': '
            if(self._name):
                delta = deltas[self._name]
                op_str += str(delta(left, right))
            else:
                op_str += str(deltas(left, right))
        return op_str

    def __repr__(self):
        op_str = ''
        if(self._name):
            op_str += str(self._name)
            op_str += ': '
        if(self._left >= 0):
            op_str += str(self._left)
        else:
            op_str += '-'
        op_str += ' vs. '
        if(self._right >= 0):
            op_str += str(self._right)
        else:
            op_str += '-'
        return op_str

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Tuple) and self._name == other._name and self._left == other._left and self._right == other._right

class Alignment(list):
    """ Models a list of tuples.
    """
    def __init__(self):
        list.__init__(self, [])

    def append_tuple(self, left, right, op = None):
        """ Appends a new tuple to the current trace.

        Args:
        left: the left index.
        right: the right index.
        op: (optional) a name for the underlying edit operation.
        """
        self.append(Tuple(op, left, right))

    def cost(self, x, y, deltas):
        """ Computes the cost of this trace. This is equivalent to
        the sum of the cost of all tuples in this trace.

        Args:
        x: A symbol list for the left indices.
        y: A symbol list for the right indices.
        deltas: A map which contains for any tuple name
                a function which assigns costs to symbol pairings.
                If provided, the cost for any oepration is rendered as
                well.

        Returns: The cost assigned by deltas to this trace.
        """
        d = 0.
        for op in self:
            d += op.cost(x, y, deltas)
        return d

    def render(self, x, y, deltas = None):
        """ Represents this trace as a string, showing the left
        and right indices in addition to the respective labels in x and y,
        and in addition to the tuple cost. This is equivalent as to
        calling 'render' on all tuples in this trace and joining the
        resulting strings with newlines.

        Args:
        x: A symbol list for the left indices.
        y: A symbol list for the right indices.
        deltas: (optional) A map which contains for any tuple name
                a function which assigns costs to symbol pairings.
                If provided, the cost for any oepration is rendered as
                well.

        Returns: A string representing this trace.
        """
        render =  []
        for op in self:
            render.append(op.render(x, y, deltas))
        return '\n'.join(render)
