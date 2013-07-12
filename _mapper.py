# -*- coding: utf-8 -*-
'''
This file is part of the Python Mapper package, an open source tool
for exploration, analysis and visualization of data.

Copyright 2011–2013 by the authors:
    Daniel Müllner, http://math.stanford.edu/~muellner
    Aravindakshan Babu, anounceofpractice@hotmail.com

Python Mapper is distributed under the GPLv3 license. See the project home page

    http://math.stanford.edu/~muellner/mapper

for more information.

-------------------------------------------------------------------------------

Implementation of the Mapper algorithm with the following characteristics:
    - for a multi-dimensional filter function
    - memory-efficient for subsets of M{R^n}, or for general metric spaces
(Memory efficiency for Euclidean data can still be improved, with an
implementation of hierarchical clustering which accepts Euclidean data instead
of a distance matrix.)
'''
from __future__ import print_function
import numpy as np
from scipy.spatial.distance import pdist
from multiprocessing import cpu_count
from threading import Thread
import sys
if sys.hexversion < 0x03000000:
    from Queue import Queue
    range = xrange
else:
    from queue import Queue
try:
    from fastcluster import linkage
except ImportError:
    sys.stderr.write('Mapper warning: Could not load the module '
                     '“fastcluster”.\nThe module “scipy.cluster.hierarchy“ is '
                     'used instead, but it will be slower.\n')
    from scipy.cluster.hierarchy import linkage

from mapper.mapper_output import mapper_output, fcluster

__all__ = ['mapper', 'single_linkage', 'complete_linkage',
           'average_linkage', 'weighted_linkage', 'centroid_linkage',
           'median_linkage', 'ward_linkage', 'n_obs', 'crop',
           'mask_data']

class single_linkage:
  '''
  Helper class. Wraps a call to single linkage clustering and provides a
  readable description.
  '''
  def __call__(self, X):
    return linkage(X, method='single')
  def __str__(self):
    return 'Single linkage clustering'

class complete_linkage:
  '''
  Helper class. Wraps a call to complete linkage clustering and provides a
  readable description.
  '''
  def __call__(self, X):
    return linkage(X, method='complete')
  def __str__(self):
    return 'Complete linkage clustering'

class average_linkage:
  '''
  Helper class. Wraps a call to average linkage clustering and provides a
  readable description.
  '''
  def __call__(self, X):
    return linkage(X, method='average')
  def __str__(self):
    return 'Average linkage clustering'

class weighted_linkage:
  '''
  Helper class. Wraps a call to weighted linkage clustering and provides a
  readable  description.
  '''
  def __call__(self, X):
    return linkage(X, method='weighted')
  def __str__(self):
    return 'Weighted linkage clustering'

class centroid_linkage:
  '''
  Helper class. Wraps a call to centroid linkage clustering and provides a
  readable description.
  '''
  def __call__(self, X):
    return linkage(X, method='centroid')
  def __str__(self):
    return 'Centroid linkage clustering'

class median_linkage:
  '''
  Helper class. Wraps a call to median linkage clustering and provides a
  readable description.
  '''
  def __call__(self, X):
    return linkage(X, method='median')
  def __str__(self):
    return 'Median linkage clustering'

class ward_linkage:
  '''
  Helper class. Wraps a call to Ward linkage clustering and provides a
  readable description.
  '''
  def __call__(self, X):
    return linkage(X, method='ward')
  def __str__(self):
    return 'Ward linkage clustering'

cluster_default = single_linkage()

def Mapper_step(q, pcd, N, point_labels, filt, cover, cluster, cutoff, M,
                metricpar,
                verbose):
    if verbose:
        print ('Start Mapper thread.')
    while True:
        level = q.get()
        if level is None: # Sentinel: end the thread
            break

        # Select the points in this filter range
        idx = cover.data_index(level)
        num_points = idx.size

        # Handle special cases.
        # 0 points in the filter interval: just skip the loop iteration.
        if num_points == 0:
            if verbose:
                print('Warning! Filter level {0} is empty.'.\
                          format(level.index))
            num_clust = 0
            Z = None
            R = None
        # 1 point => 1 cluster
        elif num_points == 1:
            if verbose:
                print('Warning! Filter level {0} has only one point.'.\
                          format(level.index))
            num_clust = 1
            points_clusters = np.zeros(1,dtype=int)
            # We label clusters starting with 0.
            Z = np.empty((0,4))
            R = 0.
        # 2 or more points: general case
        else:
            if verbose:
                print('Filter level {0} has {1} points.'.\
                          format(level.index, num_points))
            if pcd.ndim==1:
                part_data = compressed_submatrix(pcd,idx)
            else:
                part_data = pdist(pcd[idx,:], **metricpar)
            # diameter
            R = part_data.max()
            Z = cluster(part_data)
            if Z[-1,2]>R:
                print('Warning: last clustering distance is bigger than the '
                      'diameter of the filter slice ({0}>{1}).'.\
                          format(Z[-1,2], R))
                R = Z[-1,2]

            if cutoff:
                # heights in the clustering tree
                heights = Z[:,2]
                # determine a cutoff value
                # To do: Improve this!
                num_clust = cutoff(heights, R)
                # actual clustering, after the cutoff value has been determined
                points_clusters = fcluster(Z, num_clust)
                # My fcluster starts labelling clusters at 0!
                #assert num_clust == points_clusters.max()
                assert np.all(np.unique(points_clusters)==\
                                  np.arange(num_clust))

        if cutoff:
            #
            # Determine the nodes of the output graph
            #
            # Each cluster in the partial clustering gives a node
            for cl in range(num_clust):
                points = idx[ points_clusters == cl ] # This gives us the
                # indices of the clusters points in the d matrix.
                # The color is determined by the first filter component!
                attribute = np.median(filt[points,0])
                # Color the nodes by their median filter value
                #
                # Normally, the data points are labeled 0,1,...
                # Allow relabeling of the data point, whatever this is good
                # for.
                # To do: ask Aravind.
                if point_labels is not None:
                    points = point_labels[ points ]
                M.add_node( level.index, points, attribute )
        else:
            # save data for the scale graph algorithm
            M.scale_graph_data.append(dataidx=idx,
                                      dendrogram=Z,
                                      diameter=R,
                                      levelindex=level.index)


def mapper(pcd, filt, cover, cutoff,
           mask=None,
           cluster=cluster_default,
           point_labels=None,
           metricpar={},
           simple=False,
           filter_info=None,
           verbose=True):
    '''
    Mapper algorithm

    @param pcd: input data, point cloud in M{R^n}, or compressed distance
    matrix
    @type pcd: C{numpy.ndarray((N,n), dtype=float)} or
    C{numpy.ndarray((N*(N-1)/2), dtype=float)}
    @param filt: filter function with M{comp} components
    @type filt: C{numpy.ndarray((N, comp), dtype=float)} or C{numpy.ndarray(N,
    dtype=float)}
    @param cover: Class for the cover of the filter range. See L{covers}..
    @type cover: iterator
    @param cutoff: Cutoff function for the partial clustering tree. See
    L{cluster_cutoff}.
    @type cutoff: function or C{None}
    @param cluster: Clustering function.
    @type cluster: See L{single_linkage}
    @param point_labels: Labels for the input points (optional). If this is
    None, the points are labeled 0,1,...,N-1.
    @type point_labels: C{numpy.ndarray(N)}
    @param verbose: Print status message?
    @type verbose: bool

    @return: Mapper output data structure
    @rtype: mapper_output
    '''
    # input checks
    assert isinstance(pcd, np.ndarray)
    assert pcd.dtype == np.float
    if pcd.ndim==1:
        N = n_obs(pcd)
        print('Number of observations: {0}.'.format(N))
        if not np.all(pcd>=0):
            raise ValueError('Mapper needs nonnegative dissimilarity values.')
    else:
        assert pcd.ndim == 2
        [N, n] = pcd.shape
    assert isinstance(filt, np.ndarray)
    assert filt.dtype == np.float
    if filt.ndim == 1:
        filt = filt[:,np.newaxis]
    assert filt.ndim == 2
    assert np.alen(filt) == N
    if point_labels is not None:
        assert isinstance(point_labels, np.ndarray)
        assert point_labels.size == N

    # Initialize variables
    M = mapper_output(point_labels=point_labels)

    cores = cpu_count()

    if verbose:
        print('Number of CPU cores present: {0}'.format(cores))
    work_queue = Queue()
    threads = [Thread(target=Mapper_step,
                      args=(work_queue, pcd, N, point_labels, filt,
                            cover, cluster, cutoff, M, metricpar, verbose))
              for i in range(cores)]
    for t in threads: t.start()

    if verbose:
        for i in range(filt.shape[1]):
            print('Mapper: Filter range in dimension {0}: [{1:0.2f}, '
                  '{2:0.2f}]'.format(i, np.min(filt[:,i]), np.max(filt[:,i])))
        print('Mapper: Cover: {0}'.format(cover))
        print('Mapper: Clustering: {0}'.format(cluster))
        print('Mapper: Cutoff: {0}'.format(cutoff))

    # Mapper main loop
    patches = cover(filt, mask)
    if not cutoff: M.reserve_scale_graph(len(patches))
    for level in patches:
        if verbose:
            print("Level: " + str(level.index))
        M.add_level(level.index, level.range_min, level.range_max)
        work_queue.put(level)
    for i in range(cores): # Sentinels; tell the thread to stop
        work_queue.put(None)
    for t in threads:
        t.join()
    assert work_queue.empty(), ('Work qeue is not empty. Probably there was '
                                'an error in one of the parallel Mapper '
                                'threads.')

    if cutoff:
        # When all nodes have been added, make the data structure consistent
        # since we didn't keep track of the nodes in each level.
        M.add_nodes_to_levelsets()
        M.complex_from_nodes(cover=cover, verbose=verbose, simple=simple)

    # Add info
    filt_mask = filt if mask is None else filt[mask]
    if filter_info is not None:
        M.add_info(filter_info=filter_info)
    M.add_info(filter_min=np.min(filt_mask[:,0]),
               filter_max=np.max(filt_mask[:,0]))
    M.add_info(filter_min_array=np.min(filt_mask, axis=0),
               filter_max_array=np.max(filt_mask, axis=0))
    M.add_info(cover = cover.info)
    M.add_info(cutoff = str(cutoff))
    M.add_info(cluster = str(cluster))

    return M

def n_obs(dm):
    '''
    Determine the number of observations from a compressed distance matrix.

    @param dm: compressed distance matrix
    @type dm: numpy.ndarray(N*(N-1)/2, dtype=float)

    @return: M{N}, the number of observations
    @rtype: nonnegative integer
    '''
    k = np.alen(dm)
    if k==0:
        return 1
    else:
        N = int(np.ceil(np.sqrt(k * 2)))
        assert k == N*(N-1)//2
        return N

#def compressed_idx(N,r,c):
#    '''
#    The linear index of the distance from point M{r} to point M{c>r} in
#    a compressed distance matrix with M{N} data points.
#    '''
#    assert r<c
#    return (2*N-3-r)*r/2-1+c

def mask_data(data, mask, labels=None):
    if mask is None or np.all(mask):
        return data, None
    else:
        if labels is None:
            newlabels = np.flatnonzero(mask)
        else:
            newlabels = labels[mask]
        if data.ndim==1:
            return compressed_submatrix(data, np.flatnonzero(mask)), newlabels
        else:
            return data[mask], newlabels

def compressed_submatrix(dm, idx):
    '''
    Extract from a compressed distance matrix the corresponding matrix for
    a subset of points without bringing the matrix into square form first.

    The indices in the list C{idx} must be in increasing order.

    @param dm: compressed distance matrix
    @type dm: numpy.ndarray(N*(N-1)/2, dtype=float)
    @param idx: indices of the subset
    @type idx: numpy.ndarray(n, dtype=int)
    @param N: the number of observation in C{dm} (optional)
    @type N: integer

    @return: compressed distance matrix
    @rtype: numpy.ndarray(n*(n-1)/2, dtype=float)
    '''
    N = n_obs(dm)
    n = np.alen(idx)
    res = np.empty(n*(n-1)//2,dtype=dm.dtype)
    # Shorter Python code, does the same thing.
    # Which variant is faster?
    #
    #for i,c in enumerate(combinations(idx,2)):
    #    res[i] = dm[compressed_idx(N,*c)]
    for r in range(n-1):
        s = (2*n-1-r)*r//2
        t = idx[r]
        i = idx[r+1:] + (2*N-3-t)*t//2-1
        res[s:s+n-1-r] = dm[i]

    return res

def crop(f, a, b):
    from scipy.stats import scoreatpercentile
    s1 = scoreatpercentile(f, a)
    s2 = scoreatpercentile(f, 100-b)
    assert s1<=s2
    return np.logical_and(f>=s1, f<=s2)

if __name__=='__main__':
    '''Test equvalence of the Python and the C++ implementation'''
    import cmappertools
    import numpy as np

    for i in range(10000):
        N = np.random.random_integers(1000)
        n = np.random.random_integers(N)
        dm = np.random.rand(N*(N-1)//2)
        idx = np.unique(np.random.randint(N,size=n))
        r = compressed_submatrix(dm,idx)
        s = cmappertools.compressed_submatrix(dm,idx)
        if np.any(r!=s): raise AssertionError
        print("Iteration {0}: OK.".format(i))
else:
    '''Load the C++ routines, if available.'''
    try:
        from cmappertools import compressed_submatrix
    except ImportError:
        sys.stderr.write("The 'cmappertools' module could not be imported.\n")
del sys
