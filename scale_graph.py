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
'''

import sys
import math
import numpy as np
from itertools import chain, product, takewhile, count
from operator import itemgetter
if sys.hexversion < 0x03000000:
    range = xrange
from mapper.tools import progressreporter

__all__ = ['scale_graph', 'do_scale_graph']

def weighting_function(weighting):
    if weighting=='linear':
        return lambda x, y: (0,-x/y)
    elif weighting=='root':
        return lambda x, y: (0,-math.pow(x/y,.5))
    elif weighting=='inverse':
        def wf(x, y):
            if x==0.:
                return (1,0.) # infinity
            else:
                return (0,y/x)
        return wf
    elif weighting=='log':
        def wf(x, y):
            if x==0.:
                return (1,0.) # infinity
            else:
                return (0,math.log(y/x))
        return wf
    else:
        raise ValueError("Weighting '{0}' is unknown.".format(weighting))

def scale_graph(M, filt, cover=None, simple=False, **kwargs):
    do_scale_graph(M, **kwargs)
    M.nodes_from_path(filt)
    M.complex_from_nodes(cover=cover, simple=simple)

def do_scale_graph(M, weighting='inverse', exponent=0., maxcluster=None,
                   strategy=1, verbose=True, callback=None):
    '''
    Compute the scale graph from a Mapper output.
    '''
    M.add_info(cutoff="Scale graph algorithm ({0}, '{1}', {2})".\
                   format(exponent, strategy, maxcluster))

    sgd = M.scale_graph_data
    sgd.maxcluster = maxcluster

    if strategy==1:
        sgd.expand_intervals = False
    elif strategy==2:
        sgd.expand_intervals = True
    else:
        raise ValueError("Scale graph strategy '{0}' is unknown.".\
                             format(strategy))
    sgd.strategy = strategy

    dendrogram = sgd.dendrogram
    diameter = sgd.diameter
    layers = len(dendrogram)

    # Add edges
    if verbose:
        sys.stdout.write('Add edges:')
        sys.stdout.flush()

    N2, LB2, UB2, diam2 = sgd.layerdata(0)

    Dijkstra = Layered_Dijkstra(weighting=weighting)
    Dijkstra.start(N2)

    progress = progressreporter(callback)

    for i in range(1,layers):
        N1, LB1, UB1, diam1 = N2, LB2, UB2, diam2
        N2, LB2, UB2, diam2 = sgd.layerdata(i)

        Dijkstra.next_layer(N2)
        Dijkstra.add_edge(0,0)
        for j in takewhile(lambda j: LB1[j]>=diam2, range(N1)):
            Dijkstra.add_edge(j+1,0)
        for j in takewhile(lambda j: LB2[j]>=diam1, range(N2)):
            Dijkstra.add_edge(0,j+1)

        if N1 and N2:
            s0 = ( N1 if maxcluster is None else min(N1,maxcluster) ) + 1
            t0 = ( N2 if maxcluster is None else min(N2,maxcluster) ) + 1
            startk = 1
            for j in range(1, s0):
                a = LB1[j]
                b = UB1[j-1]
                for k in range(startk, t0):
                    c = LB2[k]
                    d = UB2[k-1]
                    if c>b:
                        startk += 1
                        continue
                    if d<a: break
                    maxac = max(a,c)
                    overlap = min(b,d)-maxac
                    assert overlap>=0.
                    if maxac>0:
                        Dijkstra.add_edge(j, k,
                                          overlap, np.power(maxac, exponent))
        progress(i*100//(layers-1))

    if verbose:
        print(' {0} edges in total.'.format(Dijkstra.num_edges()))

    sgd.path, sgd.infmin = Dijkstra.shortest_path()
    if verbose:
        print('Scale graph path:\n{0}'.format(sgd.path))
    sgd.edges = Dijkstra.edges

inf_fin = np.dtype([('inf',np.int),('finite',np.float)])

class Layered_Dijkstra():
    '''Caution: The weighting function may return values only with infinite
    component <= 1. Otherwise, the algorithm will not work correctly. (If
    higher infinity values are needed at some point, review the line

        self.currweight['inf'].fill(self.layer+2)

    in the method next_layer to initialize the array with even higher infinity
    values than the edge weights that can occur.)
    '''
    def __init__(self, weighting):
        self.wf = weighting_function(weighting)
        self.parent = []
        self.layer = -1
        self.edges = []

    def start(self, n):
        self.currweight = np.zeros(n+1,dtype=inf_fin)

    def next_layer(self, n):
        self.layer += 1
        self.oldweight = self.currweight
        self.currweight = np.zeros(n+1, dtype=inf_fin)
        self.currweight['inf'].fill(self.layer+2) # !!!
        self.parent.append(np.empty(n+1, dtype=np.int))
        self.parent[-1].fill(-1)
        self.edges.append([])

    def add_edge(self, s, t, w0=0., w1=1.):
        W0, W1 = self.wf(w0, w1)
        self.edges[self.layer].append((s,t,W0))
        oldw = self.oldweight[s]
        tmp = (oldw[0] + W0, oldw[1] + W1)
        # <= : prefer finer clustering in case of ties!
        if tmp <= tuple(self.currweight[t]):
            self.currweight[t] = tmp
            self.parent[self.layer][t] = s

    def num_edges(self):
        return sum(map(len,self.edges))

    def shortest_path(self):
        if not np.all(np.isfinite(self.currweight['finite'])):
            raise AssertionError
        path = np.empty(self.layer+2, dtype=np.int)
        infmin = np.min(self.currweight['inf'])
        print('Number of infinite weight edges in the path: {0}'.\
                  format(infmin))
        # Reverse array: Prefer finer paths in case of ties
        minidx = np.nonzero(self.currweight['inf']==infmin)[0][::-1]
        finite = self.currweight['finite'][minidx]
        m = np.argmin(finite)
        path[-1] = minidx[m]
        for i in range(self.layer,-1,-1):
            path[i] = self.parent[i][path[i+1]]
            if path[i]==-1:
                raise AssertionError('No path can be found.')
        return path, infmin
