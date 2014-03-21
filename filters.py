# -*- coding: utf-8 -*-
r'''
..
  This file is part of the Python Mapper package, an open source tool
  for exploration, analysis and visualization of data.

  Copyright 2011–2014 by the authors:
      Daniel Müllner, http://danifold.net
      Aravindakshan Babu, anounceofpractice@hotmail.com

  Python Mapper is distributed under the GPLv3 license. See the project
  home page

      http://danifold.net/mapper

  for more information.

  -----------------------------------------------------------------------------

Filter functions
================

Mathematical definition
-----------------------

A filter function is a function on the data set, :math:`f\:{:}\;X\to \mathbb{R}^k`. The Mapper algorithm supports general, vector-valued functions, while the GUI is restricted to real-valued functions (the case :math:`k=1`) for simplicity.

Data structure
--------------

The filter values are stored in a floating-point NumPy array of shape :math:`(n,k)`, where :math:`n` is the number of points and :math:`k` the dimensionality of the filter values.

Filter functions in Python Mapper
---------------------------------

A number of one-dimensional filter functions is provided in the module ``mapper.filters``.

The argument *data* must be a NumPy array of dimension 1 or 2. If it is one-dimensional, it is interpreted as a compressed matrix of pairwise dissimilarities (i.e. the flattened, upper part of a symmetric, quadratic matrix with zeros on the diagonal). Accordingly, this array must have length :math:`\tbinom n2` for :math:`n` data points. If the array *data* is two-dimensional of shape :math:`(n,d)`, the rows are interpreted as  :math:`n` data points in a :math:`d`-dimensional vector space, and the pairwise distances are generated from the vector data and the *metricpar* parameter. See the function :scipy:`scipy.spatial.distance.pdist` for possible entries in the *metricpar* dictionary. For example,

::

    metricpar={'metric': 'euclidean'}

generates Euclidean distances from vector data.

(In principle, there could also be filter functions which require vector data and do not work on a dissimilarity matrix. No such filter function is currently present in the module.)

The argument *callback* is used by the GUI to display progress report. Some filter functions report from time to time what percentage of the data has been processed. For example,

::

  callback=print

would just print the percentages.

The return value of the function below is always a NumPy array of shape :math:`(n,)` with data type ``float``.

'''
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.spatial.distance import squareform, cdist, pdist
import sys
import math
if sys.hexversion < 0x03000000:
    from itertools import izip as zip
    range = xrange

from mapper import n_obs, cmappertoolserror
from mapper.tools import progressreporter

__all__ = ['eccentricity', 'Gauss_density', 'kNN_distance',
           'distance_to_measure', 'graph_Laplacian', 'dm_eigenvector',
           'zero_filter']

def eccentricity(data, exponent=1.,  metricpar={}, callback=None):
    if data.ndim==1:
        assert metricpar=={}, 'No optional parameter is allowed for a dissimilarity matrix.'
        ds = squareform(data, force='tomatrix')
        if exponent in (np.inf, 'Inf', 'inf'):
            return ds.max(axis=0)
        elif exponent==1.:
            ds = np.power(ds, exponent)
            return ds.sum(axis=0)/float(np.alen(ds))
        else:
            ds = np.power(ds, exponent)
            return np.power(ds.sum(axis=0)/float(np.alen(ds)), 1./exponent)
    else:
        progress = progressreporter(callback)
        N = np.alen(data)
        ecc = np.empty(N)
        if exponent in (np.inf, 'Inf', 'inf'):
            for i in range(N):
                ecc[i] = cdist(data[(i,),:], data, **metricpar).max()
                progress((i+1)*100//N)
        elif exponent==1.:
            for i in range(N):
                ecc[i] = cdist(data[(i,),:], data, **metricpar).sum()/float(N)
                progress((i+1)*100//N)
        else:
            for i in range(N):
                dsum = np.power(cdist(data[(i,),:], data, **metricpar),
                                exponent).sum()
                ecc[i] = np.power(dsum/float(N), 1./exponent)
                progress((i+1)*100//N)
        return ecc

def Gauss_density(data, sigma, metricpar={}, callback=None):
    denom = -2.*sigma*sigma
    if data.ndim==1:
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        ds = squareform(data, force='tomatrix')
        dd = np.exp(ds*ds/denom)

        # no normalization since the dimensionality is not known
        #dd = 1/(N*(sqrt(2*pi)*sigma)^n)*exp(-ds*ds/(2*sigma*sigma)),
        # where N=#samples, n=dimensionality

        dens = dd.sum(axis=0)
    else:
        progress = progressreporter(callback)
        N = np.alen(data)
        dens = np.empty(N)
        for i in range(N):
            d = cdist(data[(i,),:], data, **metricpar)
            dens[i] = np.exp(d*d/denom).sum()
            progress(((i+1)*100//N))
        dens /= N*np.power(np.sqrt(2*np.pi)*sigma,data.shape[1])
    return dens

def kNN_distance(data, k, metricpar={}, callback=None):
    r'''The distance to the :math:`k`-th nearest neighbor as an (inverse) measure of density.

Note how the number of nearest neighbors is understood: :math:`k=1`, the first neighbor, makes no sense for a filter function since the first nearest neighbor of a data point is always the point itself, and hence this filter function is constantly zero. The parameter :math:`k=2` measures the distance from :math:`x_i` to the nearest data point other than  :math:`x_i` itself.
    '''
    if data.ndim==1:
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        # dm data
        ds = squareform(data, force='tomatrix')
        N = np.alen(ds)
        r = np.empty(N)
        for i in range(N):
            s = np.sort(ds[i,:])
            assert s[0]==0.
            r[i] = s[k]
        return r
    else:
        # vector data
        if metricpar=={} or metricpar['metric']=='euclidean':
            from scipy.spatial import cKDTree
            T = cKDTree(data)
            d, j = T.query(data, k+1)
            return d[:,k]
        else:
            print(metricpar)
            raise ValueError('Not implemented')

def distance_to_measure(data, k, metricpar={}, callback=None):
    r'''.. math::

  \mathit{distance\_to\_measure}(x)  = \sqrt{\frac 1k\sum^k_{j=1}d(x,\nu_j(x))^2},

where :math:`\nu_1(x),\ldots,\nu_k(x)` are the :math:`k`  nearest neighbors of :math:`x` in the data set. Again, the first nearest neighbor is :math:`x` itself with distance 0.

Reference: [R4]_.
'''
    if data.ndim==1:
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        # dm data
        ds = squareform(data, force='tomatrix')
        N = np.alen(ds)
        r = np.empty(N)
        for i in range(N):
            s = np.sort(ds[i,:])
            assert s[0]==0.
            d = s[1:k]
            r[i] = np.sqrt((d*d).sum()/float(k))
        return r
    else:
        # vector data
        if metricpar=={} or metricpar['metric']=='euclidean':
            from scipy.spatial import cKDTree
            T = cKDTree(data)
            d, j = T.query(data, k+1)
            d = d[:,1:k]
            return np.sqrt((d*d).sum(axis=1)/k)
        else:
            print(kwargs)
            raise ValueError('Not implemented')

def graph_Laplacian(data, eps, n=1, k=1, weighted_edges=False, sigma_eps=1.,
                    normalized=True,
                    metricpar={}, verbose=True,
                    callback=None):
    r'''Graph Laplacian of the neighborhood graph.

* First, if *k* is 1, form the *eps*-neighborhood graph of the data set: vertices are the data points; two points are connected if their distance is at most *eps*.

* Alternatively, if *k* is greater than 1, form the neighborhood graph from the
  :math:`k`-th nearest neighbors of each point. Each point counts as its first
  nearest neighbor, so feasible values start with :math:`k=2`.

* If *weighted_edges* is ``False``, each edge gets weight 1. Otherwise, each
  edge is weighted with

  .. math::

    \exp\left(-\frac{d^2}{2\sigma^2}\right),

  where :math:`\sigma=\mathtt{eps}\cdot\mathtt{sigma\_eps}` and :math:`d` is
  the distance between the two points.

* Form the graph Laplacian. The graph Laplacian is a self-adjoint operator on
  the real vector space spanned by the vertices and can thus be described by a
  symmetric matrix :math:`L`:

  If *normalized* is false, :math:`L` is closely related to the adjacency matrix of the graph: it has entries :math:`-w(i,j)` whenever nodes :math:`i` and :math:`j` are connected by an edge of weight :math:`w(i,j)` and zero if there is no edge. The :math:`i`-th diagonal entry holds the degree :math:`\deg(i)` of the corresponding vertex, so that row and column sums are zero.

  If *normalized* is true, each row :math:`i` of :math:`L` is additionally scaled by :math:`1/\sqrt{\deg(i)}`, and so is each column. This destroys the zero row and column sums but preserves symmetry.

* Return the :math:`n`-th eigenvector of the graph Laplacian. The index is 0-based: the 0-th eigenvector is constant on all vertices, corresponding to the eigenvalue 0. :math:`n=1` returns the Fiedler vector, which is the second smallest eigenvector after 0.

The normalized variant seems to give consistently better results, so this is always chosen in the GUI. However, this experience is based on few examples only, so please do not hesitate to also try the non-normalized version if there is a reason for it.

Reference: [R9]_; see especially Section 6.3 for normalization.'''
    assert n>=1, 'The rank of the eigenvector must be positive.'
    assert isinstance(k, int)
    assert k>=1
    if data.ndim==1:
        # dissimilarity matrix
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        D = data
        N = n_obs(D)
    else:
        # vector data
        D = pdist(data, **metricpar)
        N = len(data)
    if callback:
        callback('Computing: neighborhood graph.')
    rowstart, targets, weights = \
        neighborhood_graph(D, k, eps, diagonal=True,
                           verbose=verbose, callback=callback)

    c = ncomp(rowstart, targets)
    if (c>1):
        print('The neighborhood graph has {0} components. Return zero values.'.
              format(c))
        return zero_filter(data)

    weights = Laplacian(rowstart, targets, weights, weighted_edges,
                        eps, sigma_eps, normalized)

    L = scipy.sparse.csr_matrix((weights, targets, rowstart))
    del weights, targets, rowstart

    if callback:
        callback('Computing: eigenvectors.')

    assert n<N, ('The rank of the eigenvector must be smaller than the number '
                 'of data points.')

    if hasattr(spla, 'eigsh'):
        w, v = spla.eigsh(L, k=n+1, which='SA')
    else: # for SciPy < 0.9.0
        w, v = spla.eigen_symmetric(L, k=n+1, which='SA')
    # Strange: computing more eigenvectors seems faster.
    #w, v = spla.eigsh(L, k=n+1, sigma=0., which='LM')
    if verbose:
        print('Eigenvalues: {0}.'.format(w))
    order = np.argsort(w)
    if w[order[0]]<0 and w[order[1]]<abs(w[order[0]]):
        raise RuntimeError('Negative eigenvalue of the graph Laplacian found: {0}'.format(w))

    return v[:,order[n]]

def dm_eigenvector(data, k=0, mean_center=True,
        metricpar={}, verbose=True, callback=None):
    r'''Return the :math:`k`-th eigenvector of the distance matrix.

The matrix of pairwise distances is symmetric, so it has an orthonormal basis of eigenvectors. The parameter :math:`k` can be either an integer or an array of integers (for multi-dimensional filter functions). The index is zero-based, and eigenvalues are sorted by absolute value, so :math:`k=0` returns the eigenvector corresponding to the largest eigenvalue in magnitude.

If `mean_center` is ``True``, the distance matrix is double-mean-centered before the eigenvalue decomposition.

Reference: [R6]_, subsection “Principal metric SVD filters”.
    '''
    # comp can be an integer or a list of integers
    # todo: check validity of comp
    if data.ndim==1:
        # dissimilarity matrix
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        D = data
        N = n_obs(D)
    else:
        # vector data
        D = pdist(data, **metricpar)
        N = len(data)

    DD = squareform(D)
    del D

    if mean_center:
        md = DD.mean(axis=1)
        DD -= md
        DD -= (md-md.mean())[:,np.newaxis]

    karray = np.atleast_1d(k)
    assert karray.ndim == 1
    maxk = 1 + karray.max()

    if callback:
        callback('Computing: distance matrix eigenvectors.')

    if hasattr(spla, 'eigsh'):
        w, v = spla.eigsh(DD, k=maxk, which='LM')
    else: # for SciPy < 0.9.0
        w, v = spla.eigen_symmetric(DD, k=maxk, which='LM')

    sortedorder = np.argsort(np.abs(w))[::-1]
    if verbose:
        print('Eigenvalues:\n{}'.format(w[sortedorder]))

    ret = v[:,sortedorder[k]]

    # normalize
    return ret / np.sqrt((ret*ret).sum(axis=0))

def zero_filter(data, **kwargs):
    r'''Return an array of the correct size filled with zeros.'''
    if data.ndim==1:
        return np.zeros(n_obs(data))
    else:
        return np.zeros(np.alen(data))

neighborhood_graph = cmappertoolserror('neighborhood_graph')

ncomp = cmappertoolserror('ncomp')

Laplacian = cmappertoolserror('Laplacian')

if __name__ == "__main__":
    '''Test the C++ routines against the Python code.'''
    import cmappertools
    seed = np.random.randint(0,1e10)
    print("Seed: {0}".format(seed))
    np.random.seed(seed)

    def check(M, N):
        absdiff = np.float(np.abs(M-N))
        reldiff = absdiff/np.maximum(np.abs(M), np.abs(N))
        abserr = np.max(absdiff)
        relerr = np.max(reldiff)
        if abserr>1e-13 and relerr>1e-12:
            print(M)
            print(N)
            raise AssertionError('Tolerance exceeded: abs {0}, rel {1}.'.\
                                     format(abserr, relerr))

    def test_ecc(X, n_exp=100, **kwargs):
        exponents = np.hstack((1, np.inf, np.random.rand(n_exp)*10+.001))
        try:
            for exponent in exponents:
                e1 = eccentricity(X, exponent=exponent, metricpar=kwargs)
                e2 = cmappertools.eccentricity(X, exponent=exponent,
                                               metricpar=kwargs)
                check(e1, e2)
        except AssertionError:
            sys.stderr.write('Exponent: {0}\n'.format(exponent))
            raise

    def test_Gauss(X, sigma, **kwargs):
        d1 = Gauss_density(X, sigma=sigma, metricpar=kwargs)
        d2 = cmappertools.Gauss_density(X, sigma=sigma, metricpar=kwargs)
        check(d1, d2)

    def test_dm(max_n=100):
        n = np.random.randint(2, max_n)
        N = n*(n-1)//2
        scale = np.random.rand(1)*100
        dm = np.random.rand(N)*scale
        test_ecc(dm)
        sigma = .01+np.random.rand(1)*2*scale
        test_Gauss(dm, sigma)

    def test_vector(max_n=20, max_dim=4):
        n = np.random.randint(2, max_n)
        d = np.random.randint(2, max_dim)
        X = np.random.randn(n,d)
        sigma = .01+np.random.rand(1)*2

        metric='euclidean'
        p = None
        test_ecc(X)
        test_Gauss(X, sigma=sigma)
        metric='euclidean'
        test_ecc(X, metric=metric)
        test_Gauss(X, metric=metric, sigma=sigma)
        metric='chebychev'
        test_ecc(X, metric=metric)
        test_Gauss(X, metric=metric, sigma=sigma)
        metric='minkowski'
        for p in np.hstack((1, np.random.rand(10)*10+1.)):
            test_ecc(X, metric=metric, p=p)
            test_Gauss(X, metric=metric, p=p, sigma=sigma)

    i = 0
    try:
        while True:
            test_dm(100)
            test_vector(100,20)
            i=i+1
            print('OK. '+ str(i))
    except AssertionError:
        sys.stderr.write('Seed: {0}\n'.format(seed))
        raise

else:
    '''Load the C++ routines, if available.'''
    try:
        from cmappertools import \
            eccentricity, \
            Gauss_density, \
            neighborhood_graph, \
            ncomp, \
            Laplacian
    except ImportError:
        sys.stderr.write("The 'cmappertools' module could not be imported.\n")

# Local variables:
# mode: Python
# python-indent: 4
# End:
