.. edist documentation master file, created by
   sphinx-quickstart on Fri Apr 24 14:57:39 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Edit Distances
=====================

This library contains several edit distance and alignment algorithms for
sequences and trees of arbitrary node type. Additionally, this library
contains multiple backtracing mechanisms for every algorithm in order to
facilitate more detailed interpretation and subsequent processing.

In more detail, this library currently features the following algorithms.

* Sequence Edit Distance (sed; Levenshtein, 1965)
* Dynamic Time Warping (sed; Vintsyuk, 1968)
* Affine edit distance (aed; Gotoh, 1982)
* Tree Edit Distance (ted; Zhang and Shasha, 1989)
* Constrained Unordered Tree Edit Distance (uted; Zhang and Shasha, 1996)
* Set edit distance (seted; unpublished)

As well as the following meta-algorithms:

* Algebraic Dynamic Programming (adp; according to the dissertation `Paaßen, 2019`_)
* Embedding Edit Distance Learning (bedl;  `Paaßen et al., 2018`_)

If you intend to use this library in academic work, please cite the paper:

* Paaßen, B., Mokbel, B., & Hammer, B. (2015). A Toolbox for Adaptive Sequence
    Dissimilarity Measures for Intelligent Tutoring Systems. In O. C. Santos,
    J. G. Boticario, C. Romero, M. Pechenizkiy, A. Merceron, P. Mitros,
    J. M. Luna, et al. (Eds.), Proceedings of the 8th International Conference
    on Educational Data Mining (pp. 632-632). International Educational
    Datamining Society.

Please consult the `project website <https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances/>`_ for more detailed information about the project.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   adp
   aed
   alignment
   bedl
   dtw
   edits
   multiprocess
   sed
   seted
   ted
   uted
   tree_edits
   tree_utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Paaßen, 2019: https://doi.org/10.4119/unibi/2935545
.. _Paaßen et al., 2018: http://proceedings.mlr.press/v80/paassen18a.html
