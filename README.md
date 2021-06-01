# Python Edit Distances

Copyright (C) 2019-2021 - Benjamin Paassen  
Machine Learning Research Group  
Center of Excellence Cognitive Interaction Technology (CITEC)  
Bielefeld University

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

## Introduction

This library contains several edit distance and alignment algorithms for
sequences and trees of arbitrary node type. Additionally, this library
contains multiple backtracing mechanisms for every algorithm in order to
facilitate more detailed interpretation and subsequent processing. Finally,
this library provides a reference implementation for embedding edit distance
learning (BEDL; [Paaßen et al., 2018][Paa2018]), which enables users to learn
edit distance parameters instead of specifying them manually.

Refer to the Quickstart Guide for how to use the library and refer to the
list below for a full list of the enclosed algorithms. The detailed API
documentation is available at [readthedocs.org](https://edist.readthedocs.io/en/latest/index.html).

If you use this library in academic work, please cite:

* Paaßen, B., Mokbel, B., & Hammer, B. (2015). A Toolbox for Adaptive Sequence
    Dissimilarity Measures for Intelligent Tutoring Systems. In O. C. Santos,
    J. G. Boticario, C. Romero, M. Pechenizkiy, A. Merceron, P. Mitros,
    J. M. Luna, et al. (Eds.), Proceedings of the 8th International Conference
    on Educational Data Mining (pp. 632-632). International Educational
    Datamining Society. ([Link][Paa2015])

```
@inproceedings{Paassen2015EDM,
author = {Paaßen, Benjamin and Mokbel, Bassam and Hammer, Barbara},
booktitle = {Proceedings of the 8th {International Conference on Educational Data Mining} ({EDM 2015})},
date = {2015-06},
editor = {Olga Christina Santos and Jesus Gonzalez Boticario and Cristobal Romero and Mykola Pechenizkiy and Agathe Merceron and Piotr Mitros and Jose Maria Luna and Christian Mihaescu and Pablo Moreno and Arnon Hershkovitz and Sebastian Ventura and Michel Desmarais},
language = {English},
pages = {632–632},
publisher = {International Educational Datamining Society},
title = {A Toolbox for Adaptive Sequence Dissimilarity Measures for Intelligent Tutoring Systems},
url = {http://www.educationaldatamining.org/EDM2015/uploads/papers/paper_257.pdf},
venue = {Madrid, Spain},
year = {2015}
}
```

This library is historically based on its Java version, the
[TCS Alignment Toolbox][tcs].

## Installation

This package is available on [pypi][pypi] as `edist`. You can install
it via

```
pip install edist
```

If you wish to build this project from source, you need to first install
[cython][cython] and then execute the following commands in this directory:

```
python3 cython_setup.py build_ext --inplace
cp *so edist/.
```

## Quickstart Guide

There are multiple example cases illustrated in our demo notebooks.
In particular:

* `sed_demo.ipynb` illustrates the Levenshtein distance
    (Levenshtein, 1965) and affine edit distance
    ([Gotoh, 1982][Got1982]) as well as its backtracing,
* `dtw_demo.ipynb` illustrates dynamic time warping ([Vintsyuk, 1968][Vin1968])
    as well as its backtracing and speedup measures,
* `ted_demo.ipynb` illustrates the tree edit distance
    ([Zhang and Shasha, 1989][Zha1989]) as well as its backtracing and
    support for edit functions, and
* `bedl_demo.ipynb` illustrates embedding edit distance learning
    ([Paaßen et al., 2018][Paa2018]).

In general, applying this library works as follows. First, you select the
edit distance function that best fits for your data and your setting
(see below for an overview of all available functions). Let's say your
function is called `distfun`. Then, you can compute the distance between two
lists/trees `x` and `y` via `distfun(x, y)`. If you wish to compute the matrix
of all pairwise distances for an entire dataset of lists/trees `X`, then you
can use the `multiprocess` module as follows.

```
from edist.multiprocess import pairwise_distances_symmetric
D = pairwise_distances_symmetric(X, distfun)
```

If you wish to compute the matrix of all pairwise distances between one
dataset `X` and another dataset `Y`, you can use the following function.

```
from edist.multiprocess import pairwise_distances
D = pairwise_distances(X, Y, distfun)
```

If you wish to use a custom local distance function `delta`, you can supply
it as additional argument to either `distfun` itself, to
`pairwise_distances_symmetric`, or to `pairwise_distances`.

If you wish to compute the optimal alignment between two lists/trees `x`
and `y` according to `distfun`, you can use the function
`distfun_backtrace(x, y)`. Note that, in case of multiple possible optimal
alignments, this function will always return the option that uses replacements
as early as possible. If you instead wish to sample a random optimal alignment,
you can use `distfun_backtrace_stochastic(x, y)`. Unfortunately, it is
infeasible to enumerate the entire set of co-optimal alignments because this
set may be exponentially large. However, it is possible to characterize the
distribution of co-optimal alignments concisely by describing with which
probability each node in `x` is paired with each node in `y`. This probability
matrix is computed by the function `distfun_backtrace_matrix(x, y)` and follows
the forward-backward algorithm developed by [Paaßen (2018)][Paa2018arxiv].

## List of Algorithms and Functions

The following edit distance algorithms and functions are contained in this
library.

* The [Levenshtein distance][Lev]/sequence edit distance (Levenshtein, 1965):
  * `edist.sed.standard_sed(x, y)` for edit distance computation between
    sequences `x` and `y` with a cost of 1 for each replacement, deletion,
    and insertion.
  * `edist.sed.sed_string(x, y)` for the same, but specifically designed for
    strings and thus considerably faster (~factor 3).
  * `edist.sed.standard_sed_backtrace(x, y)` for backtracing for the standard
    edit distance.
  * `edist.sed.standard_sed_backtrace_stochastic(x, y)` for the same, but
    returning a random optimal alignment instead of a fixed one.
  * `edist.sed.standard_sed_backtrace_matrix(x, y)` for the same, but
    returning a probability distribution over all pairings between elements
    of `x` and `y`.
  * `edist.sed.sed(x, y, delta)` for edit distance computation with a custom
    element distance function `delta`.
  * `edist.sed.sed_backtrace(x, y, delta)` for backtracing for the edit
    distance with a custom element distance function `delta`.
  * `edist.sed.sed_backtrace_stochastic(x, y, delta)` for the same, but
    returning a random optimal alignment instead of a fixed one.
  * `edist.sed.sed_backtrace_matrix(x, y, delta)` for the same, but
    returning a probability distribution over all pairings between elements
    of `x` and `y`.
* The [dynamic time warping][dtw] distance (DTW; [Vintsyuk, 1968][Vin1968]):
  * `edist.dtw.dtw_numeric(x, y)` for DTW computation between two time
    series `x` and `y`, each given as a double array.
  * `edist.dtw.dtw_manhattan(x, y)` for DTW computation between two time
    series `x` and `y`, each given as a double matrix. The distance between
    two frames is defined as the Manhattan distance.
  * `edist.dtw.dtw_euclidean(x, y)` for DTW computation between two time
    series `x` and `y`, each given as a double matrix. The distance between
    two frames is defined as the Euclidean distance.
  * `edist.dtw.dtw_string(x, y)` for DTW computation between two strings
    `x` and `y`.
  * `edist.dtw.dtw(x, y, delta)` for DTW computation between two arbitrary
    sequences `x` and `y` with a custom element distance function `delta`.
  * `edist.dtw.dtw_backtrace(x, y, delta)` for backtracing for DTW with a
    custom element distance function `delta`.
  * `edist.dtw.dtw_backtrace_stochastic(x, y, delta)` for the same, but
    returning a random optimal alignment instead of a fixed one.
  * `edist.dtw.dtw_backtrace_matrix(x, y, delta)` for the same, but
    returning a probability distribution over all pairings between elements
    of `x` and `y`.
* The affine edit distance ([Gotoh, 1982][Got1982]):
  * `edist.aed.aed(x, y, rep, gap, skip)` for affine edit distance computation
    between two arbitrary sequences `x` and `y`, where each frame replacement
    is scored with the function `rep`, each deletion and insertion is scored
    with the function `gap`, and each deletion and insertion extension is
    scored with the function `skip`.
  * `edist.aed.aed_backtrace(x, y, rep, gap, skip)` for backtracing for the
    affine edit distance with the replacement cost function `rep`, the
    deletion/insertion cost function `gap`, and the gap extension cost function
    `skip`.
  * `edist.aed.aed_backtrace_stochastic(x, y, delta)` for the same, but
    returning a random optimal alignment instead of a fixed one.
  * `edist.aed.aed_backtrace_matrix(x, y, delta)` for the same, but
    returning a probability distribution over all pairings between elements
    of `x` and `y`.
* The tree edit distance (TED; [Zhang and Shasha, 1989][Zha1989]):
  * `edist.ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)` for edit distance
    computation between the trees `x` and `y`, which are both given in a
    node list/adjacency list format. Both lists are supposed to be in
    depth-first-search order, e.g. a tree a(b, c) is supposed to be represented
    as the two lists `['a', 'b', 'c']` and `[[1, 2], [], []]`. The cost for
    replacements, deletions, and insertions is fixed to 1.
  * `edist.ted.standard_ted_backtrace(x_nodes, x_adj, y_nodes, y_adj)`
    for backtracing for the tree edit distance.
  * `edist.sed.standard_sed_backtrace_matrix(x_nodes, x_adj, y_nodes, y_adj)`
    for the same, but returning a probability distribution over all pairings
    between elements of `x` and `y`.
  * `edist.ted.ted(x_nodes, x_adj, y_nodes, delta)` for tree edit distance
    computation with a custom node distance function `delta`.
  * `edist.ted.ted_backtrace(x_nodes, x_adj, y_nodes, delta)` for backtracing
    for the tree edit distance with a custom element distance function `delta`.
  * `edist.ted.ted_backtrace_matrix(x_nodes, x_adj, y_nodes, delta)` for the
    same, but returning a probability distribution over all pairings between
    elements of `x` and `y`.
* The unordered tree edit distance (UTED; [Zhang, 1996][Zha1996]):
    * `edist.uted.uted(x_nodes, x_adj, y_nodes, y_adj, delta)` for edit
    distance computation between the trees `x` and `y`, which are both given
    in a node list/adjacency list format. Both lists are supposed to be in
    depth-first-search order, e.g. a tree a(b, c) is supposed to be represented
    as the two lists `['a', 'b', 'c']` and `[[1, 2], [], []]`.
    * `edist.uted.uted_backtrace(x_nodes, x_adj, y_nodes, y_adj, delta)` for
    backtracing of the unordered tree edit distance with an (optional) custom
    element distance function `delta`.
* The set edit distance (SetED; unpublished, but using the Hungarian algorithm
    of [Kuhn, 1955][Kuh1955] at its core):
  * `edist.seted.standard_seted(x, y)` for set edit distance computation
    between the sets `x` and `y`, which are both given as lists for
    convenience. The cost for replacements, deletions, and insertions is fixed
    to 1.
  * `edist.seted.standard_seted_backtrace(x, y)` for backtracing for the set
    edit distance.
  * `edist.seted.seted(x, y, delta)` for set edit distance computation with a
    custom element distance function `delta`.
  * `edist.seted.seted_backtrace(x, y, delta)` for backtracing for the set
    edit distance with a custom element distance function `delta`.

Additionally, this library contains a few helper modules, namely:

* `edist.adp` contains functions to compute arbitrary sequence edit distances
    that can be defined by a regular grammar. This is based on the framework
    of algebraic dynamic programming (ADP; [Giegerich, Meyer, and Steffen, 2004][Gie2004]),
    as applied by Paaßen, Mokbel, and Hammer ([2016][Paa2016]). In particular:
  * `edist.adp.edit_distance(x, y, grammar, deltas)` computes the sequence edit
    distance defined by the regular grammar `grammar` and the cost functions
    `deltas` between sequences `x` and `y`,
  * `edist.adp.backtrace(x, y, grammar, deltas)` computes the backtracing
    for said edit distance,
  * `edist.adp.backtrace_stochastic(x, y, grammar, deltas)` does the same,
    but returns a random optimal alignment instead of a fixed one, and
  * `edist.adp.backtrace_matrix(x, y, grammar, deltas)` does the same,
    but returns a probability distribution over all pairings between elements
    of `x` and `y`.
* `edist.alignment` models backtraces/alignments between sequences or trees.
    Instances of class `edist.alignment.Alignment` are returned by every
    backtrace function (except for `backtrace_matrix` functions).
* `edist.bedl` supports embedding edit distance learning (BEDL;
    [Paaßen et al., 2018][Paa2018]) to learn parameters for edit distance
    instead of learning them manually. Please refer to the `bedl_demo` for
    more information.
* `edist.edits` supports objects that model sequence edits, in particular
    replacements, deletions, and insertions, and provides the function
    `alignment_to_script(alignment, x, y)`, which transforms the alignment
    `alignment` between the sequences `x` and `y` into a list of edits that
    transform `x` into `y`.
* `edist.tree_edits` supports objects that model tree edits, in particular
    replacements, deletions, and insertions, and provides the function
    `alignment_to_script(alignment, x_nodes, y_adj, y_nodes, y_adj)`, which
    transforms the alignment `alignment` between the trees `x` and `y` into a
    list of edits that transform `x` into `y`.
* `edist.tree_utils` is a collection of supporting functions for tree handling
    used by the library. Interesting for users of the library may be the
    following:
  * `edist.tree_utils.to_json` writes a tree to a JSON file.
  * `edist.tree_utils.from_json` reads a tree from a JSON file.
  * `edist.tree_utils.dataset_from_json` reads a list of trees from a
    directory containing JSON files.
  * `edist.tree_utils.tree_to_string` formats a tree as a string.

## Background

The background for all algorithms covered in this library by far exceeds the
scope of this Readme (we list material for further reading below).
However, there are a few general points that are worth noting in short:

* Edit distance algorithms heavily rely on _dynamic programming_ to be
  efficient, i.e. to decompose the overall edit distance computation into
  subtasks and to tabulate the results of these subtasks. In this way, we
  can search an exponentially large space of possible alignments in polynomial
  time. However, these decompositions rely on the critical assumption that
  the element distance function $`\delta`$ is a metric, especially that this
  function is non-negative, zero for self-distances, and fulfills the
  triangular inequality. If any of these assumptions is broken, there may
  exist cheaper alignments that are not covered by the dynamic programming
  computation. So this is critical when generating your own $`\delta`$
  functions.
* Interestingly, if $`\delta`$ is a metric, its metric properties translate
  to the overall edit distance (refer, e.g., to Theorem 3.2. in
  [Paaßen, 2019][Paa2019]). This can make handling edit distances quite
  appealing, mathematically. However, this does _not_ hold for dynamic time
  warping, which always violates the triangular inequality.
* Even though dynamic programming makes edit distances polynomial, computing
  them can become prohibitively expensive for long sequences/large trees. In
  particular, any sequence edit distance lies in $`\mathcal{O}(m \cdot n)`$,
  where $`m`$ and $`n`$ are the lengths of the input sequences, the tree
  edit distance lies in $`\mathcal{O}(m^2 \cdot n^2)`$, and the set edit
  distance in $`\mathcal{O}((m+n)^3)`$. Fortunately, the [cython][cython]
  implementation provided in this library is relatively fast and thus can cope
  with $`m, n`$ even up to a few thousand elements (at least for sequence
  edits). Still, it is key that you choose the edit distance function that is
  best fitting to your case. For example, `edist.sed.sed_string` is about
  factor 15 faster compared to the more general `edist.sed.sed`.

For more background on the algorithms, we refer to the Wikipedia articles for
the [Levenshtein distance][Lev] and [dynamic time warping][dtw], to the paper
of [Gotoh (1982)][Got1982] with respect to the affine edit distance, to the
review paper of [Paaßen (2018)][Paa2018] with respect to the tree edit distance
and its backtracing, to Section 2.3.2 of the dissertation of [Paaßen (2019)][Paa2019]
with respect to algebraic dynamic programming, and to Chapter 4 of the same
dissertation with respect to embedding edit distance learning.

## Licensing

This library is licensed under the [GNU General Public License Version 3][GPLv3].

## Dependencies

This library depends on [NumPy][np] for matrix operations, and on [cython][cython]
for the effective C-interface. Further, the `bedl.py` module depends on
[scikit-learn][scikit] for the base interfaces and on [SciPy][scipy] for
optimization. Finally, the `seted.pyx` module depends on [SciPy][scipy] for
an implementation of the Hungarian algorithm ([Kuhn, 1955][Kuh1955]).

## Literature

* Giegerich, R., Meyer, C., & Steffen, P. (2004). A discipline of dynamic
    programming over sequence data. Science of Computer Programming, 51(3),
    215-263. doi:[10.1016/j.scico.2003.12.005][Gie2004]
* Gotoh, O. (1982). An improved algorithm for matching biological sequences.
    Journal of Molecular Biology, 162(3), 705-708.
    doi:[10.1016/0022-2836(82)90398-9][Got1982]
* Kuhn, H. (1955). The Hungarian method for the assignment problem.
    Naval Research Logistics Quarterly, 2(1-2), 83-97.
    doi:[10.1002/nav.3800020109][Kuh1955]
* Levenshtein, V. (1965). Binary codes capable of correcting deletions,
    insertions, and reversals. Soviet Physics Doklady, 10(8), 707-710.
* Paaßen, B., Mokbel, B., & Hammer, B. (2015). A Toolbox for Adaptive Sequence
    Dissimilarity Measures for Intelligent Tutoring Systems. In O. C. Santos,
    J. G. Boticario, C. Romero, M. Pechenizkiy, A. Merceron, P. Mitros,
    J. M. Luna, et al. (Eds.), Proceedings of the 8th International Conference
    on Educational Data Mining (pp. 632-632). International Educational
    Datamining Society. [Link][Paa2015]
* Paaßen, B., Mokbel, B., & Hammer, B. (2016). Adaptive structure metrics for
    automated feedback provision in intelligent tutoring systems.
    Neurocomputing, 192, 3-13.
    doi:[10.1016/j.neucom.2015.12.108](https://doi.org/10.1016/j.neucom.2015.12.108).
    [Link][Paa2016]
* Paaßen, B., Gallicchio, C., Micheli, A., & Hammer, B. (2018). Tree Edit
    Distance Learning via Adaptive Symbol Embeddings. Proceedings of the 35th
    International Conference on Machine Learning (ICML 2018), 3973-3982.
    [Link][Paa2018]
* Paaßen, B. (2018). Revisiting the tree edit distance and its backtracing:
    A tutorial. arXiv:[1805.06869][Paa2018arxiv].
* Paaßen, B. (2019). Metric Learning for Structured Data. Dissertation.
    Bielefeld University. doi:[10.4119/unibi/2935545][Paa2019]
* Vintsyuk, T.K. (1968). Speech discrimination by dynamic programming.
    Cybernetics, 4(1), 52-57. doi:[10.1007/BF01074755][Vin1968]
* Zhang, K., & Shasha, D. (1989). Simple Fast Algorithms for the Editing
    Distance between Trees and Related Problems. SIAM Journal on Computing,
    18(6), 1245-1262. doi:[10.1137/0218082][Zha1989]
* Zhang, K. (1996). A Constrained Edit Distance Between Unordered Labeled
    Trees. Algorithmica, 15, 205-222. doi:[10.1007/BF01975866][Zha1996]

<!-- References -->

[Paa2015]:http://www.educationaldatamining.org/EDM2015/uploads/papers/paper_257.pdf "Paaßen, B., Mokbel, B., & Hammer, B. (2015). A Toolbox for Adaptive Sequence Dissimilarity Measures for Intelligent Tutoring Systems. In O. C. Santos, J. G. Boticario, C. Romero, M. Pechenizkiy, A. Merceron, P. Mitros, J. M. Luna, et al. (Eds.), Proceedings of the 8th International Conference on Educational Data Mining (pp. 632-632). International Educational Datamining Society."
[Paa2016]:https://pub.uni-bielefeld.de/record/2783224 "Paaßen, B., Mokbel, B., & Hammer, B. (2016). Adaptive structure metrics for automated feedback provision in intelligent tutoring systems. Neurocomputing, 192, 3-13. doi:10.1016/j.neucom.2015.12.108"
[Paa2018]:http://proceedings.mlr.press/v80/paassen18a.html "Paaßen, B., Gallicchio, C., Micheli, A., & Hammer, B. (2018). Tree Edit Distance Learning via Adaptive Symbol Embeddings. Proceedings of the 35th International Conference on Machine Learning (ICML 2018), 3973-3982."
[Paa2018arxiv]:https://arxiv.org/abs/1805.06869 "Paaßen, B. (2018). Revisiting the tree edit distance and its backtracing: A tutorial. arXiv:1805.06869."
[Paa2019]:https://doi.org/10.4119/unibi/2935545 "Paaßen, B. (2019). Metric Learning for Structured Data. Dissertation. Bielefeld University. doi:10.4119/unibi/2935545"
[Lev]:https://en.wikipedia.org/wiki/Levenshtein_distance "Wikipedia page on Levenshtein distance."
[dtw]:https://en.wikipedia.org/wiki/Dynamic_time_warping "Wikipedia page on dynamic time warping."
[Vin1968]:https://doi.org/10.1007/BF01074755 "Vintsyuk, T.K. (1968). Speech discrimination by dynamic programming. Cybernetics, 4(1), 52-57. doi:10.1007/BF01074755"
[Got1982]:https://doi.org/10.1016/0022-2836(82)90398-9 "Gotoh, O. (1982). An improved algorithm for matching biological sequences. Journal of Molecular Biology, 162(3), 705-708. doi:10.1016/0022-2836(82)90398-9"
[Gie2004]:https://doi.org/10.1016/j.scico.2003.12.005 "Giegerich, R., Meyer, C., & Steffen, P. (2004). A discipline of dynamic programming over sequence data. Science of Computer Programming, 51(3), 215-263. doi:10.1016/j.scico.2003.12.005"
[Zha1989]:https://doi.org/10.1137/0218082 "Zhang, K., & Shasha, D. (1989). Simple Fast Algorithms for the Editing Distance between Trees and Related Problems. SIAM Journal on Computing, 18(6), 1245-1262. doi:10.1137/0218082"
[Zha1996]:https://doi.org/10.1007/BF01975866 "Zhang, K. (1996). A Constrained Edit Distance Between Unordered Labeled Trees. Algorithmica, 15, 205-222. doi:10.1007/BF01975866"
[Kuh1955]:https://doi.org/10.1002/nav.3800020109 "Kuhn, H. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2), 83-97. doi:10.1002/nav.3800020109"
[tcs]:https://openresearch.cit-ec.de/projects/tcs "TCS Alignment Toolbox homepage"
[pypi]:https://pypi.org/project/edist/ "PyPi edist project page"
[cython]:https://cython.org/ "cython homepage"
[scikit]: https://scikit-learn.org/stable/ "Scikit-learn homepage"
[np]: http://numpy.org/ "Numpy homepage"
[scipy]: https://scipy.org/ "SciPy homepage"
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html "The GNU General Public License Version 3"
