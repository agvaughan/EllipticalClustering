
The files included here are used to generate tests of Random Mixed Selectivity in the following paper:

*Structured and cell-type-specific encoding of decision variables in orbitofrontal cortex*

Junya Hirokawa, Alexander Vaughan, Paul Masset Torben Ott and Adam Kepecs

These routines test the assumption that a set of random n-dimensional response vectors constitute an elliptical Gaussian distribution.  In the context of this paper, the response vectors are n-dimensional coefficients corresponding to neuronal response profiles after PCA.  IFF these coefficients are distributed as an elliptical Gaussian, the underlying neuronal representations can be said to show Random Mixed Selectivity.

The ePAIRS test, a derivative of the PAIRS test developed by Matt Kaufman in [1], examines the distribution of distances arising from comparing each response vectors to its nearest neighbor.  If response vectors are organized into a small number of clusters, this average distance will be smaller than that of random vectors. 

The eRP test, a derivative of the Random Projection test developed by Brian Lau in https://github.com/brian-lau/highdim, examines the distribution of distances arising from each response vector to a random n-dimensional vector.  If response vectors are organized into a small set of clusters, this average distance wil be *larger* than that of random vectors.

The major modifications to the PAIRS and RP tests address the issue of elliptical distributions - which cause false positives for both tests even for elliptical gaussians.  Because an elliptical gaussian distribution is *not* spherically uniform after scaling each axis to sphericity, it is important to develop a set of expected distances for PAIRS and RP that arise from a similarly sized elliptical distribution instead.

For the eRP test, we also modified the statistical test to use an explicit bootstrap test; we found that the previous approach was unstable and had low specificity.

We thank previous authors for their contribution to this work, and have endeavored to leave the original authorship and licensing intact as much as possible.



[1] Raposo, David, Matthew T. Kaufman, and Anne K. Churchland. "A category-free neural population supports evolving demands during decision-making." Nature neuroscience 17.12 (2014): 1784.
