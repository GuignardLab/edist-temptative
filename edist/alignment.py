"""
Implements an alignment between two sequences or trees.

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

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class Tuple:
    """ Models a single alignment entry with an edit operation name,
    a left index, and a right index.

    Attributes
    ----------
    _name: str
        The name of the corresponding edit operation.
    _left: int
        The index of the aligned object on the left or -1 if no such object
        exists.
    _right: int
        The index of the aligned object on the right or -1 if no such object
        exists.

    """
    def __init__(self, name, left, right):
        self._name = name
        self._left = left
        self._right = right

    def cost(self, x, y, deltas):
        """ Computes the cost of the current edit tuple.

        Parameters
        ----------
        x: list
            A symbol list for the left indices.
        y: list
            A symbol list for the right indices.
        deltas: function or dictionary
            The cost function delta mapping pairs of elements to
            replacement/deletion/insertion costs OR
            A map which contains for any operation name such a function.

        Returns
        -------
        cost: float
            The cost assigned by deltas to this tuple.

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

        Parameters
        ----------
        x: list
            A symbol list for the left indices.
        y: list
            A symbol list for the right indices.
        deltas: function or dictionary (default = None)
            The cost function delta mapping pairs of elements to
            replacement/deletion/insertion costs OR
            A map which contains for any operation name such a function.
            If provided, the cost for any operation is rendered as well.

        Returns
        -------
        repr: str
            A string representing this tuple.

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
    """ Models a list of tuples. Note that, by convention, an alignment between
    sequences only permits tuples in lexicographically ascending order, i.e.
    an alignment of nothing to 0, 0 to 1, and 1 to 2, should be stored in that
    order and not as [(1, 2), (0, 1), (-1, 0)], for example. The same holds for
    tree alignments, with the additional requirement that aligned indices must
    respect the structure of the tree, i.e. if i is aligned to j and i2 to j2,
    then i can only be a parent of i2 if j is a parent of j2 (and vice versa).

    """
    def __init__(self):
        list.__init__(self, [])

    def append_tuple(self, left, right, op = None):
        """ Appends a new tuple to the current Alignment.

        Parameters
        ----------
        left: int
            the left index.
        right: int
            the right index.
        op: str (default = None)
            a name for the underlying edit operation.

        """
        self.append(Tuple(op, left, right))

    def cost(self, x, y, deltas):
        """ Computes the cost of this trace. This is equivalent to
        the sum of the cost of all tuples in this trace.

        Parameters
        ----------
        x: list
            A symbol list for the left indices.
        y: list
            A symbol list for the right indices.
        deltas: function or dictionary
            The cost function delta mapping pairs of elements to
            replacement/deletion/insertion costs OR
            A map which contains for any operation name such a function.

        Returns
        -------
        cost: float
            The cost assigned by deltas to this Alignment.

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

        Parameters
        ----------
        x: list
            A symbol list for the left indices.
        y: list
            A symbol list for the right indices.
        deltas: function or dictionary (default = None)
            The cost function delta mapping pairs of elements to
            replacement/deletion/insertion costs OR
            A map which contains for any operation name such a function.
            If provided, the cost for any operation is rendered as well.

        Returns
        -------
        repr: str
            A string representing this Alignment.

        """
        render =  []
        for op in self:
            render.append(op.render(x, y, deltas))
        return '\n'.join(render)
