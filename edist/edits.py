"""
Implements list edits, i.e. functions which take a list as input and
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

import abc
import copy

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class Edit(abc.ABC):
    """ An abstract parent class for all edits.

    """
    @abc.abstractmethod
    def apply(self, lst):
        """ Applies this edit to the given list and returns a copy of the
        list with the applied changes. The original list remains unchanged.

        Parameters
        ----------
        lst: list
            a list

        Returns
        -------
        res: list
            a copy of the list with the applied edit.

        """
        pass

    @abc.abstractmethod
    def apply_in_place(self, lst):
        """ Applies this edit to the given list. Note that this changes the
        input argument.

        Parameters
        ----------
        lst: list
            a list

        """
        pass


class Replacement(Edit):
    """ Replaces node self._index with label self._label.

    Attributes
    ----------
    _index: int
        The index of the node to be deleted.
    _label: str
        The new label for node self._index.

    """
    def __init__(self, index, label):
        self._index = index
        self._label = label

    def apply(self, lst):
        """ Replaces entry self._index with self._label.

        Parameters
        ----------
        lst: list
            a list

        Returns
        -------
        res: list
            a copy of the list with the applied edit.

        """
        lst = copy.copy(lst)
        self.apply_in_place(lst)
        return lst

    def apply_in_place(self, lst):
        """ Replaces entry self._index with self._label.

        Parameters
        ----------
        lst: list
            a list

        """
        lst[self._index] = self._label

    def __repr__(self):
        return 'rep(%d, %s)' % (self._index, str(self._label))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Replacement) and self._index == other._index and self._label == other._label


class Deletion(Edit):
    """ Deletes node self._index.

    Attributes
    ----------
    _index: int
        The index of the node to be deleted.

    """
    def __init__(self, index):
        self._index = index

    def apply(self, lst):
        """ Deletes the self._indexth entry.

        Parameters
        ----------
        lst: list
            a list

        Returns
        -------
        res: list
            a copy of the list with the applied edit.

        """
        lst = copy.copy(lst)
        self.apply_in_place(lst)
        return lst

    def apply_in_place(self, lst):
        """ Deletes the self._indexth entry.

        Parameters
        ----------
        lst: list
            a list

        """
        # delete the nodes entry at index
        del lst[self._index]

    def __repr__(self):
        return 'del(%d)' % (self._index)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Deletion) and self._index == other._index


class Insertion(Edit):
    """ Inserts a new self._label entry at position self._index .

    Attributes
    ----------
    _index: int
        The index at which this edit will be applied.
    _label: str
        The new entry.

    """
    def __init__(self, index, label):
        self._index = index
        self._label = label

    def apply(self, lst):
        """ Inserts a new self._label entry at position self._index.

        Parameters
        ----------
        lst: list
            a list

        Returns
        -------
        res: list
            a copy of the list with the applied edit.

        """
        lst = copy.copy(lst)
        self.apply_in_place(lst)
        return lst

    def apply_in_place(self, lst):
        """ Inserts a new self._label entry at position self._index.

        Parameters
        ----------
        lst: list
            a list

        """
        lst.insert(self._index, self._label)

    def __repr__(self):
        return 'ins(%d, %s)' % (self._index, str(self._label))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Insertion) and self._index == other._index and self._label == other._label


class Script(list, Edit):
    """ A list of Edits.

    """
    def __init__(self, lst = []):
        list.__init__(self, lst)

    def apply(self, lst):
        """ Applies all edits in this script.

        Parameters
        ----------
        lst: list
            a list

        Returns
        -------
        res: list
            a copy of the list with the applied edits of this script.

        """
        # if this script is empty, just return the input
        if(not self):
            return lst
        # otherwise make a copy and then apply the script in place
        lst = copy.copy(lst)
        self.apply_in_place(lst)
        return lst

    def apply_in_place(self, lst):
        """ Applies all edits in this script in place.

        Parameters
        ----------
        lst: list
            a list

        """
        for e in range(len(self)):
            self[e].apply_in_place(lst)


def alignment_to_script(alignment, x, y):
    """ Converts the given alignment into an edit script which maps the given
    list x to the given list y such that all entries of the alignment are
    translated one to one into edits.

    Note that the order of operations does change because the script will
    first apply replacements (in input order), then deletions
    (in descending order), and finally insertions (in ascending order),
    which simplifies conversion.

    Parameters
    ----------
    alignment: class alignment.Alignment
        an Alignment object which maps between the given lists x and y.
    x: list
        a list.
    y: list
        another list.

    Returns
    -------
    script: class edits.Script
        A Script object script such that script.apply(x) = y and where every
        tuple in the alignment has one corresponding edit.

    """
    script = Script()
    # iterate over the alignment and sort replacements, deletions, and
    # insertions
    dels = []
    inss = []
    for op in alignment:
        if(op._left >= 0):
            if(op._right >= 0):
                # if both left and right are >= 0, we have a replacement,
                # which we can process right away
                # We ignore replacements that change nothing, though.
                if(x[op._left] != y[op._right]):
                    script.append(Replacement(op._left, y[op._right]))
            else:
                # if only the left index is defined, we have a deletion, which
                # we process later
                dels.append(op._left)
        else:
            if(op._right < 0):
                raise ValueError('Invalid tuple in alignment: %s.' % str(op))
            # if only the right index is defined, we have an insertion, which
            # we process later
            inss.append(op._right)
    # apply deletions in reverse order
    for i in sorted(dels, reverse=True):
        script.append(Deletion(i))
    # apply insertions in ascending order
    for j in sorted(inss):
        script.append(Insertion(j, y[j]))
    # return the script
    return script
