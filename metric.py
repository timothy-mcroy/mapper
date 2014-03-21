# -*- coding: utf-8 -*-
'''
This file is part of the Python Mapper package, an open source tool
for exploration, analysis and visualization of data.

Copyright 2011–2014 by the authors:
    Daniel Müllner, http://danifold.net
    Aravindakshan Babu, anounceofpractice@hotmail.com

Python Mapper is distributed under the GPLv3 license. See the project home page

    http://danifold.net/mapper

for more information.
'''
'''
Intrinsic metric for data sets
'''
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import sys
from mapper import n_obs, cmappertoolserror
from mapper.tools import progressreporter

# More imports below

def intrinsic_metric(data, k=1, eps=0.,
                     metricpar={}, allow_disconnected=False,
                     verbose=True, callback=None):
    r'''Intrinsic metric.'''
    data = np.array(data)
    if data.ndim==1:
        # dissimilarity matrix
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        D = data
    else:
        # vector data
        D = pdist(data, **metricpar)
    G = neighborhood_graph(D, k, eps, verbose=verbose, callback=callback)

    if not allow_disconnected:
        c = ncomp(*G)
        if c>1:
            raise AssertionError('The neighborhood graph is disconnected. '
                                 'It has {0} components.'.format(c))
    return graph_distance(*G, callback=callback)

class _csr_graph:
    def __init__(self, N):
        self.N = N
        self.degrees = np.zeros(N, dtype=np.intp)

    def reserve_edge(self, r):
        self.degrees[r] += 1

    def finalize(self):
        self.rowstart = np.empty(self.N+1, dtype=np.intp)
        self.rowstart[0] = 0
        self.rowstart[1:] = np.cumsum(self.degrees)
        self.nextedge = self.rowstart.copy()
        self.targets = np.empty(self.rowstart[-1], dtype=np.intp)
        self.weights = np.empty(self.rowstart[-1])
        del self.degrees

    def add_edge(self, r, c, w):
        i = self.nextedge[r]
        self.targets[i] = c
        self.weights[i] = w
        self.nextedge[r] += 1
        assert self.nextedge[r]<=self.rowstart[r+1]

    def validate(self):
        for i in range(self.N):
            assert self.nextedge[i]==self.rowstart[i+1]
        del self.nextedge

    def __str__(self):
        s = ('CSR graph\n'
             'Number of vertices: {0}\n'
             'Number of edges; {1}\n').format(self.N, len(self.targets))
        r = 0
        for i, (c, w) in enumerate(zip(self.targets, self.weights)):
            if i==self.rowstart[r+1]:
                r += 1
            s += str((r,c,w)) + '\n'
        return s

def neighborhood_graph(D, k=1, eps=0., verbose=True, callback=None):
    r'''Neighborhood graph.'''
    ds = squareform(D, force='tomatrix')

    np.fill_diagonal(ds, np.inf)
    min_eps = ds.min(axis=0).max()
    if verbose:
        print('Minimal epsilon: {0}.'.format(min_eps))
    N = len(ds)

    G = _csr_graph(N)
    UF = union_find(N)

    i = 0
    for r in range(N):
        for c in range(r+1,N):
            if D[i]<=eps:
                G.reserve_edge(r)
                G.reserve_edge(c)
                UF.Union(r,c)
            i += 1
    row1, col1 = np.nonzero(ds<=eps)

    if k>1:
        d, j = nearest_neighbors_from_dm(D, k, callback=callback)
        del d

        # Add two edges (both directions) for each pair of kNN
        row_col  = np.empty((2*N*k,2), dtype=j.dtype)
        row_col[:N*k,0] = np.repeat(np.arange(N), k)
        row_col[:N*k,1] = j.flat # np.ravel(j) ?
        row_col[N*k:,0] = row_col[:N*k,1]
        row_col[N*k:,1] = row_col[:N*k,0]

        # Remove points as their own nearest neighbors.
        row_col = row_col[row_col[:,0]!=row_col[:,1]]

        # Keep only edges with distance >eps
        s = (ds[row_col[:,0],row_col[:,1]] > eps)
        row_col = row_col[s]

        # Remove all duplicate edges.
        row, col = _unique_rows(row_col).T # TBD timeit

        for r in row:
            G.reserve_edge(r)

        for r,c in zip(row,col):
            UF.Union(r,c)

    G.finalize()

    i = 0
    for r in range(N):
        for c in range(r+1,N):
            if D[i]<=eps:
                G.add_edge(r, c, D[i])
                G.add_edge(c, r, D[i])
            i += 1

    if k>1:
        for r,c in zip(row,col):
            G.add_edge(r,c,ds[r,c])

    G.validate()

    if UF.ncomp>1:
        print('The neighborhood graph has {0} components.'.format(UF.ncomp))

    G.ncomp = UF.ncomp
    return (G.rowstart, G.targets, G.weights)

# From http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
def _unique_rows(row_col):
    u = np.unique(row_col.view(dtype=[('',(row_col.dtype,row_col.shape[-1]))]))
    return u.view(dtype=row_col.dtype).reshape((len(u),row_col.shape[-1]))

def nearest_neighbors_from_dm(X, k, callback=None):
    ''' This is inefficient. To be done:

        (1) Do not fully sort every row of the distance matrix but find the
        first k=lo elements.

        (2) Use the compressed distance matrix, not the square form.

        Both improvements are realized in cmappertools 1.0.5.
    '''
    progress = progressreporter(callback)
    D = squareform(X, force='tomatrix')
    N = np.alen(D)
    j = np.empty((N,k), dtype=np.int32)
    d = np.empty((N,k))
    for i, row in enumerate(D):
        j[i] = np.argsort(row)[:k]
        d[i] = D[i,j[i]]
        progress((i+1)*100//N)
    return d, j

def dm_from_data(X, **kwargs):
    print('Warning: Inefficient conversion from vector to dm data!')
    from scipy.spatial.distance import pdist
    return pdist(X, **kwargs)

def _conn_comp_loop(j):
    N = np.alen(j)
    UF = union_find(N)
    for kk in xrange(1,j.shape[1]):
        for jj in xrange(N):
            UF.Union(jj,j[jj,kk])
        if UF.ncomp==1: break
    return UF.ncomp, kk

def minimal_k_to_make_dataset_connected(data, lo=10, metricpar={},
                                            callback=None, verbose=True):
    if data.ndim==1:
        # dissimilarity matrix
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.', metricpar)
        D = data
        N = n_obs(D)

        def nn(k):
            return nearest_neighbors_from_dm(D, k, callback)
    elif metricpar['metric']=='euclidean':
        # vector data, Euclidean metric
        data_cKDTree = cKDTree(data)
        N = len(data)

        def nn(k):
            return data_cKDTree.query(data,k)
    else:
        # vector data
        print("Inefficient! Generate pairwise distances from vector data.")
        D = pdist(data, **metricpar)
        N = len(data)

        def nn(k):
            return nearest_neighbors_from_dm(D, k, callback)

    k = lo
    ncomp = 2
    while ncomp>1:
        if verbose:
            print('Try up to {0} neighbors.'.format(k))
        d, j = nn(k)
        assert np.all(d[:,0]==0.)
        if verbose:
            print('Compute threshold for connectedness.')
        ncomp, kk = _conn_comp_loop(j)
        if k==N: assert ncomp==1
        k = min(2*k,N)

    return kk+1

class union_find:
    def __init__(self, size):
        self.parent = np.empty(size, dtype=np.int)
        self.parent.fill(-1)
        self.sizes = np.ones(size, dtype=np.int)
        self.ncomp = size

    def Find(self, idx):
        if (self.parent[idx] != -1 ): # a → b
            p = idx
            idx = self.parent[idx]
            if (self.parent[idx] != -1 ): # a → b → c
                while True:
                    idx = self.parent[idx];
                    if self.parent[idx]==-1: break
                while True:
                    self.parent[p], p = idx, self.parent[p]
                    if self.parent[p]==idx: break
        return idx

    def Union(self, node1, node2):
        node1 = self.Find(node1)
        node2 = self.Find(node2)
        if node1==node2: return
        if self.sizes[node1]<self.sizes[node2]:
            self.parent[node1] = node2
            self.sizes[node2] += self.sizes[node1]
        else:
            self.parent[node2] = node1
            self.sizes[node1] += self.sizes[node2]
        self.ncomp -= 1

cmappertools_version = cmappertoolserror('cmappertools_version')

ncomp = cmappertoolserror('ncomp')

graph_distance = cmappertoolserror('graph_distance')

if __name__!='__main__':
    try:
        from cmappertools import \
            __version__ as cmappertools_version, \
            neighborhood_graph, \
            ncomp, \
            graph_distance, \
            _conn_comp_loop, \
            nearest_neighbors_from_dm
        print('Using cmappertools v' + cmappertools_version + '.')
    except ImportError:
        sys.stderr.write('Intrinsic metric is not available.\n')
else:
    '''Test neighborhood_graph from cmappertools against the Python version.'''
    import cmappertools

    seed = np.random.randint(10000000000)
    print("Seed: {0}".format(seed))
    np.random.seed(seed)

    for i in range(100):
        print("Test {0}/100:".format(i+1))
        N = np.random.randint(20,200)
        NN = N*(N-1)//2
        D = np.random.rand(NN)
        eps = np.random.rand(1)
        k = np.random.randint(1,N)

        G = neighborhood_graph(D, k, eps)
        H = cmappertools.neighborhood_graph(D, k, eps)

        D = cmappertools.graph_distance(*G)
        E = cmappertools.graph_distance(*H)

        assert np.all(D==E), np.max(np.abs(D-E))
    print("OK.")

# Local variables:
# mode: Python
# End:
