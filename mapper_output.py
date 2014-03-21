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
This module contains all the classes to define the L{mapper_output} class. As the name says, it is used as a container for the output from the Mapper algorithm, and the methods for displaying and analyzing the Mapper output accept this class as the input format.
'''
import sys
from itertools import chain, combinations, count
# Efficient iterators for both Python 2 and 3
if sys.hexversion < 0x03000000:
    from itertools import izip as zip
    range = xrange

from collections import defaultdict
from pickle import load, dump
from numpy import ndarray, intersect1d, zeros
#from dionysus import Simplex
import scipy.sparse as scsp

import numpy as np

from mapper.tools import dict_items, dict_values
from mapper.draw_mapper_output import draw_2D, draw_scale_graph, save_scale_graph_as_pdf

# Make the database imports optional
try:
    import tools.sql.psycopg2_config
    from tools.sql.psycopg2_aux import iterable_to_table
    import mapper_results_from_db
except ImportError:
    pass

class mapper_output:
    '''
    Mapper output class

    This class stores the information given by the Mapper algorithm
    in the following attributes:

      - B{info}: a dictionary. This is a freeform attribute. Use it to store
        any information like the data source, the Mapper parameters, the
        algorithm details (clustering method), etc.

      - B{levelsets}: a dictionary of L{levelset} objects. The index of
        each levelset is the key in the dictionary.

      - B{nodes}: a list of L{node} objects. These are the vertices in the
        simplicial complex.

      - B{simplices}: the higher-dimensional simplices in the Mapper output,
        stored as a L{simpl_complex} object. Only the simplices of dimension 1
        and higher are stored here since the vertices are stored with some
        extra information as the nodes.

      - vertex_pos
    '''
    def __init__(self, info=None, levelsets=None, nodes=None, simplices={},
                 point_labels=None):
        '''
        Generate a new L{mapper_output} object.

        The input format for simplices is a dictionary of
        C{(tuple of nodes): attribute}. It is converted to a L{simpl_complex}
        structure on generation.

        @param info: info dictionary
        @type info: dict
        @param levelsets: level sets
        @type levelsets: dictionary with L{levelset} values
        @param nodes: list of nodes
        @type nodes:  list of L{node} objects
        @param simplices: all simplices other than vertices
        @type simplices: dict
        @param point_labels: point labels
        @type point_labels: numpy array
        '''
        # input checks
        if info is None: info = {}
        if levelsets is None: levelsets = {}
        if nodes is None: nodes = []

        assert isinstance(info, dict)
        assert isinstance(levelsets, dict)
        for x in levelsets.values():
            assert isinstance(x, levelset)
        assert isinstance(nodes, list)
        for x in nodes:
            assert isinstance(x, node)

        self.info = info
        self.levelsets = levelsets
        self.nodes = nodes
        self.simplices = simpl_complex(simplices)
        self.point_labels = point_labels
        self.scale_graph_data = scale_graph_data()

    def __str__(self):
        return 'Mapper output.' + \
            '\nInfo     : '      + str(self.info) + \
            '\nLevelsets: ' + str(self.levelsets) + \
            '\nNodes    : '     + str(self.nodes) + \
            '\nSimplices: ' + str(self.simplices)

    def __repr__(self):
        '''
        Represent the data of the Mapper output by a string.
        '''
        return repr({
            'info':      self.info,
            'levelsets': self.levelsets,
            'nodes':     self.nodes,
            'simplices': self.simplices
        })

    def add_info(self, **kwargs):
        '''
        Add a key-value pair to the C{info} dictionary.
        '''
        self.info.update(kwargs)

    def add_node(self, level, points, attribute=None):
        '''
        Add a node.

        See L{node.__init__} for the input parameters. A running index is
        automatically generated.
        '''
        self.nodes.append(node(level, points, attribute))

    def add_level(self, level, filter_min, filter_max, nodes=None):
        '''
        Add a level set.

        See L{levelset.__init__} for the input parameters.

        There are two modes to generate consistent data: Either the list of
        node indices in each level set is given when the level set is created,
        or the method L{add_nodes_to_levelsets} is called when all nodes have
        been added.
        '''
        # Note to self: the default nodes=[] since Python would
        # use a shallow copy of the same empty list for each
        # level set.
        assert level not in self.levelsets
        self.levelsets[level] = levelset(filter_min, filter_max, nodes)

    def add_nodes_to_levelsets(self):
        '''
        Complete the list of nodes in each levelset, in case this
        has not been kept track of.
        '''
        for i, node in enumerate(self.nodes):
            self.levelsets[node.level].nodes.add(i)

    def add_simplex(self, vertices, weight=1):
        '''
        Add a simplex of dimension S{>=}1 to the simplicial complex.

        See L{simpl_complex.add_simplex} for the input parameters.
        '''
        #try:
        #    dionysus_simplex = isinstance(vertices, Simplex)
        #except NameError:
        #    dionysus_simplex = False
        #if dionysus_simplex:
        #    assert weight==1
        #    weight = vertices.data
        #    vertices = vertices.vertices
        #else:
        #    assert isinstance(vertices,(list,tuple))
        assert isinstance(vertices, (list,tuple))
        if len(vertices) > 0:
            self.simplices.add_simplex(vertices, weight)
        else:
            raise ValueError('Empty lists cannot be added as simplices.')

    def generate_complex(self, cover=None, verbose=False, min_sizes=(),
                         max_dim = -1):
        '''
        Generate the simplicial complex from the intersections of the point
        sets for each node.

        The weight of each simplex is the number of data points in the
        intersection.

        This is a generic algorithm which works in every case but might not be
        fast. E.g. it tests every pair of nodes for intersecting point sets,
        wheres it is often known from the patch arrangement in the cover that
        many patches do not intersect. Feel free to use a different scheme
        when speed is an issue.

        @param verbose: print progress messages?
        @type verbose: bool
        '''
        '''
        The data scheme for the dictionary S: For v1<v2<...<vn,
        S[(v1,v2,...,v(n-1)][vn] stores the data points in the intersection of
        the patches U_v1, ..., U_vn if it is nonempty. This is exactly the
        condition that (v1,...,vn) form simplex. We iteratively generate this
        data, starting from S[()][i] = (data points for the node i).
        '''
        dim = 0
        print("There are {0} nodes.".format(self.num_nodes))
        min_nodesize = 1 if len(min_sizes)<1 else min_sizes[0]
        S0 = dict()
        for i, n in enumerate(self.nodes):
            if n.points.size>=min_nodesize:
                S0[i] = n.points
                self.add_simplex((i,), len(n.points))
        S = {(): S0}

        #S = {() : dict([(i, n.points) for i, n in enumerate(self.nodes) \
        #                    if n.points.size>=min_nodesize])}
        if verbose:
            print("Generate the simplicial complex.")
        while S: # while S is not empty
            dim += 1
            if max_dim >= 0  and dim > max_dim: break
            min_simplexsize = 1 if len(min_sizes)<=dim else min_sizes[dim]
            if verbose:
                print ("Collect simplices of dimension {0}:".format(dim))
            T = defaultdict(dict)
            for i1, Si1 in dict_items(S):
                for i2, i3 in combinations(Si1,2):
                    intersection = intersect1d(Si1[i2], Si1[i3],
                                               assume_unique=True)
                    if intersection.size >= min_simplexsize:
                        if i2>i3: # ensure i2<i3
                            i2, i3 = i3, i2
                        self.add_simplex( i1 + (i2,i3),
                                          weight=intersection.size )
                        T[i1 + (i2,)][i3] = intersection
            S = T
            if verbose:
                print("There are {0} simplices of dimension {1}.".\
                          format(sum(map(len,dict_values(S))), dim) )

    def generate_complex_new(self, cover=None,
                             verbose=False, min_sizes=(), max_dim = -1):
        '''
        Generate the simplicial complex from the intersections of the point
        sets for each node.

        The weight of each simplex is the number of data points in the
        intersection.

        This is a generic algorithm which works in every case but might not be
        fast. E.g. it tests every pair of nodes for intersecting point sets,
        wheres it is often known from the patch arrangement in the cover that
        many patches do not intersect. Feel free to use a different scheme
        when speed is an issue.

        @param verbose: print progress messages?
        @type verbose: bool
        '''
        '''
        The data scheme for the dictionary S: For v1<v2<...<vn,
        S[(v1,v2,...,v(n-1)][vn] stores the data points in the intersection of
        the patches U_v1, ..., U_vn if it is nonempty. This is exactly the
        condition that (v1,...,vn) form simplex. We iteratively generate this
        data, starting from S[()][i] = (data points for the node i).
        '''
        dim = 0
        print("There are {0} nodes.".format(self.num_nodes))
        min_nodesize = 1 if len(min_sizes)<1 else min_sizes[0]
        S0 = []
        for i, n in enumerate(self.nodes):
            if n.points.size>=min_nodesize:
                S0.append((i, n.points))
                self.add_simplex((i,), len(n.points))
        S = [((), S0)]

        if verbose:
            print("Generate the simplicial complex.")
        ci = cover.cannot_intersect
        while S: # while S is not empty
            dim += 1
            if max_dim >= 0  and dim > max_dim: break
            min_simplexsize = 1 if len(min_sizes)<=dim else min_sizes[dim]
            if verbose:
                print ("Collect simplices of dimension {0}:".format(dim))
            T = []
            for i1, Si1 in S:
                l1 = tuple([self.nodes[i].level for i in i1])
                for j, (i2, Si2) in enumerate(Si1):
                    l2 = self.nodes[i2].level
                    t = []
                    for i3, Si3 in Si1[j+1:]:
                        l3 = self.nodes[i3].level
                        # Always assume that nodes from the same levelset are
                        # disjoint
                        if l2==l3 or ci(l1+(l2,l3)): continue
                        intersection = intersect1d(Si2, Si3,
                                                   assume_unique=True)
                        if intersection.size >= min_simplexsize:
                            self.add_simplex( i1+(i2,i3),
                                              weight=intersection.size )
                            t.append((i3, intersection))
                    if t:
                        T.append((i1+(i2,),t))
            S = T
            if verbose:
                print("There are {0} simplices of dimension {1}.".\
                          format(sum([len(b) for a,b in S]), dim) )

    def generate_graph(self, cover=None, verbose=False, min_sizes=()):
        '''
        Generate a graph from the intersections of the point sets for each
        node.

        The weight of each simplex is the number of data points in the
        intersection.

        The difference to the C{generate_complex} method is that only
        intersections of levelsets with consecutive indices are taken into
        account, and that no higher simplices than edges are generated. This
        difference affects the situation when the "overlap" parameter is
        50% or greater. The C{generate_complex} method creates lots of
        triangles, while the C{generate_graph} method tends to have chains of
        nodes.

        @param verbose: print progress messages?
        @type verbose: bool
        '''
        print("There are {0} nodes.".format(self.num_nodes))
        min_nodesize = 1 if len(min_sizes)<1 else min_sizes[0]
        for i, n in enumerate(self.nodes):
            assert len(n.level)==1, \
                'This method can only be used for the 1-dimensional Mapper.'
            if n.points.size>=min_nodesize:
                self.add_simplex((i,), len(n.points))

        min_simplexsize = 1 if len(min_sizes)<=1 else min_sizes[1]
        if verbose:
            print ("Collect edges:")
        for (i,), (j,) in combinations(self.simplices[0],2):
            Ni, Nj = self.nodes[i], self.nodes[j]
            if abs(Ni.level[0]-Nj.level[0])!=1: continue
            intersection_size = intersect1d(Ni.points, Nj.points,
                                            assume_unique=True).size
            if intersection_size >= min_simplexsize:
                if i>j: # ensure i<j
                    i, j = j, i
                self.add_simplex((i,j), weight=intersection_size)
        if verbose:
            print("There are {0} edges.".format(len(self.simplices[1])))

    @property
    def dimension(self):
        '''
        Convenience property: the dimension of the simplicial complex.

        Note that the dimension may change when simplices are added.

        @return: dimension
        @rtype: integer S{>=}0
        '''
        return self.simplices.dimension

    @property
    def num_nodes(self):
        '''
        Convenience property: number of nodes.

        @rtype: int
        '''
        return len(self.nodes)

    @property
    def node_sizes(self):
        '''
        Convenience property: return a dictionary of sizes of all nodes.

        @rtype: list of positive integers
        '''
        return dict([(index, len(node.points)) for index, node in enumerate(self.nodes)])

    def save_to_file(self, filename):
        '''
        Save the Mapper output to a Pickle file.

        @param filename: file name
        @type filename: str
        '''
        assert isinstance(filename, str)
        with open(filename, mode='wb') as file:
            # pickle.dump
            dump(self, file)

    @staticmethod
    def load_from_file(filename):
        '''
        Load a Mapper output structure from a Pickle file.

        @param filename: file name
        @type filename: str

        @rtype: mapper_output
        '''
        assert isinstance(filename, str)
        with open(filename, mode='rb') as file:
            # pickle.load
            M = load(file)
        return M

    def to_simple_Graph(self):
        '''
        Convert the 1-skeleton of a L{mapper_output} to a networkx Graph. The
        nodes are nonnegative integers.
        No C{info} or C{levelset} dictionary, just the graph itself.
        @rtype: C{networkx.Graph}
        '''
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.simplices[0])
        G.add_weighted_edges_from([edge + (weight,) for edge, weight in \
                                       dict_items(self.simplices[1])])
        return G

    def to_d3js_graph(self):
        """
        Convert the 1-skeleton of a L{mapper_output} to a dictionary
        designed for exporting to a json package for use with the d3.js
        force-layout graph visualisation system.
        """
        G = {}

        G['vertices'] = [{'index': i, 'level': n.level, 'members':
                          list(n.points), 'attribute': n.attribute}
                         for (i,n) in enumerate(self.nodes)]

        G['edges'] = [{'source': e[0], 'target': e[1], 'wt':
                       self.simplices[1][e]} for e in
                      self.simplices[1].keys()]

        return G

    @staticmethod
    def from_Graph(G):
        '''
        Convert a networkx Graph to a 1-dimensional L{mapper_output}. This
        only makes sense for graphs which have been converted from Mapper
        output before, since this method needs more information than just the
        graph, like a C{levelset} and an C{info} dictionary.

        @param G: graph
        @type G: C{networkx.Graph}

        @rtype: L{mapper_output}
        '''
        indexed_nodes = dict([(d['index'], v) for v, d in G.nodes(data=True)])
        sorted_nodes = [indexed_nodes[i] for i in range(G.number_of_nodes())]

        M = mapper_output(info=G.graph['info'],
                          levelsets=G.graph['levelsets'],
                          nodes=sorted_nodes,
                          simplices=dict([((G.node[e[0]]['index'], G.node[e[1]]['index']), e[2]['weight']) for e in G.edges_iter(data=True)]) )
        return M

    def adjacency_matrix(self, weighted=False, sparse=True):
        '''
        Build the (weighted or unweighted) adjacency matrix of the 1-skeleton
        of a Mapper output.

        The edge weights are the number of data points in the intersections of
        two nodes.

        By default, the adjacency matrix is output as a sparse matrix in
        "Compressed Sparse Column" format (C{scipy.sparse.csc_matrix}). This
        had no deep reason. Change the sparse format if a different one is
        more appropriate.

        @param weighted: Weighted edges? (Default: False = unweighted)
        @type weighted: bool
        @param sparse: Sparse or dense output matrix? (Default: True =
        compressed)
        @type sparse: bool

        @rtype: matrix
        '''
        assert isinstance(weighted, bool)
        assert isinstance(sparse, bool)
        dtype = int if weighted else bool
        inifn = scsp.csc_matrix if sparse else zeros
        A = inifn((self.num_nodes,self.num_nodes), dtype=dtype)
        if weighted:
            for edge, weight in dict_items(self.simplices[1]):
                A[edge] = weight
                A[edge[::-1]] = weight
        else:
            for edge in self.simplices[1]:
                A[edge] = True
                A[edge[::-1]] = True
        return A

    def to_db( self, cursor ):

        """
        Given a psycopg2 cursor object that points to a postgres db with the
        mapper_output schema, writes the mapper_output objects information to
        db.

        Returns the expr_id, which can be passed to (C{mapper_output.from_db})
        to retrieve the object.

        @param cursor: db cursor
        @type cursor: psycopg2._psycopg.cursor

        @rtype: int
        """

        # TBD: Check that this function still works after the changes to
        # levelset logic

        #assert isinstance(  cur, psycopg2._psycopg.cursor )
        assert self.info["cover"]["dim"] == 1
        levels_and_levelsets = list( dict_items(self.levelsets) )
        levels_and_levelsets.sort( key = lambda x: x[0][0] )
        nodes = self.nodes
        edges = self.simplices[1].keys( )
        # should we modify db to store weights as well?

        dataset_id = self.info["dataset_id"]
        intervals = int( self.info["cover"]["intervals"] )
        overlap = int( self.info["cover"]["fract_overlap"]*100 )
        cover = self.info["cover"]["type"]
        cutoff = self.info["cutoff"]
        cluster = self.info["cluster"]
        filter_min = float(self.info["filter_min"])
        filter_max = float(self.info["filter_max"])

        cursor.execute( """select nextval( 'seq_mapper_experiments' );""" )
        expr_id = cursor.fetchall( )[0][0]
        cursor.execute( """
                        insert into mapper_experiments( dataset_id, expr_id, intervals,
                                                        overlap, filter_min, filter_max,
                                                        cover, cutoff, cluster )
                        values( %(dataset_id)s, %(expr_id)s, %(intervals)s,
                                %(overlap)s, %(filter_min)s, %(filter_max)s,
                                %(cover)s, %(cutoff)s, %(cluster)s );
                        """, locals( ) )
        #tmp_flattened_info = tools.flatten_dict( mapper_output_info )
        #tmp = [ (expr_id, attribute, value) for attribute,value in tmp_flattened_info ]
        #iterable_to_table( cursor, tmp, "mapper_experiments_attributes" )

        # Write levels and filter values
        tmp = [ (expr_id, level[0], float(levelset.filter_min), float(levelset.filter_max)) for level, levelset in levels_and_levelsets ]
        iterable_to_table( cursor, tmp, "mapper_levels" )

        # Get node ids
        tmp = [ (expr_id, i, n.level[0], float(n.attribute)) for i,n in enumerate(nodes)]
        iterable_to_table( cursor, tmp, "mapper_nodes" )

        # Write point sets
        tmp_seq_of_point_seqs = [ [ (expr_id, node_id, point) for point in node.points ] for node_id, node in enumerate(nodes) ]
        tmp = chain( *tmp_seq_of_point_seqs )
        iterable_to_table( cursor, tmp, "mapper_points" )

        # Write edges
        if len(edges) > 0:
            tmp = [ (expr_id, u, v) for u,v in edges ]
            iterable_to_table( cursor, tmp, "mapper_edges" )

        cursor.connection.commit( )
        self.add_info( expr_id = expr_id )

        return expr_id

    @staticmethod
    def from_db( cursor, expr_id ):
        """
        Given a psycopg2 cursor object that points to a postgres db with the mapper_output schema
        and an expr_id, will retrieve the corresponding mapper_output object.

        @param cursor: db cursor
        @type cursor: psycopg2._psycopg.cursor

        @param expr_id: experiment id for the mapper experiment you want to retrieve from the db
        @type expr_id: int

        @rtype: mapper_output
        """

        expr_ids, expr_info_by_expr_ids, levelsets_by_expr_ids, nodes_by_expr_ids, edges_by_expr_ids =\
            mapper_results_from_db.mapper_results_from_db( cursor, expr_ids = [expr_id] )

        M = mapper_output(info=expr_info_by_expr_ids[0],
                          levelsets=levelsets_by_expr_ids[0],
                          nodes=nodes_by_expr_ids[0],
                          simplices=dict( [((u, v), 1.) for (u,v) in edges_by_expr_ids[0] ] ) )
        return M

    # TBD: Remove this method?
    def remove_nodes(self, nodes, verbose=False):
        '''
        Remove nodes from the Mapper output.

        @param nodes: list of nodes.
        @type nodes: list of integers
        '''
        nodes = list(nodes)
        if verbose:
            print("Cleanup: Remove the nodes {0}.".format(nodes))
        if len(nodes)==0: return
        # update the level sets
        for ls in dict_values(self.levelsets):
            ls.nodes.difference_update(nodes)
        # make a map from old node indices to new indices
        nodes.sort()
        offset = 0
        c = nodes[offset]
        node_map = [None] * self.num_nodes
        for i in range(self.num_nodes):
            if i==c:
               offset += 1
               c = nodes[offset] if len(nodes)>offset else None
            else:
               node_map[i] = i - offset
        nm = lambda x: node_map[x]
        # update the simplicial complex
        D = dict()
        for s, v in dict_items(self.simplices.as_dict()):
            if node_map[s[0]] is not None:
                D[tuple(map(nm,s))] = v
        self.simplices = simpl_complex(D)

        # update the list of nodes
        self.nodes = [self.nodes[i] for i in range(self.num_nodes) \
                          if i not in nodes ]

    def clear_nodes(self):
        self.nodes = []
        self.simplices = simpl_complex()
        for ls in dict_values(self.levelsets):
            ls.nodes = set()

    def reserve_scale_graph(self, *args):
        self.scale_graph_data.reserve(*args)

    def cutoff(self, cutoff, filt, cover=None, simple=False, verbose=True):
        self.path_from_cutoff(cutoff, verbose=verbose)
        self.nodes_from_path(filt)
        self.complex_from_nodes(cover=cover, simple=simple)

    def path_from_cutoff(self, cutoff, verbose=True):
        self.add_info(cutoff=str(cutoff))

        path = np.empty(len(self.scale_graph_data), dtype=np.int)
        for i, (idx, Z, R) in enumerate(self.scale_graph_data):
            levelindex = (i,)
            num_points = len(idx)
            if num_points<=1:
                num_clust = num_points
            else:
                heights = Z[:,2]
                num_clust = cutoff(heights, R)
            path[i] = num_clust
        if verbose:
            print('Scale graph path: {}'.format(path))
        self.scale_graph_data.path = path
        self.scale_graph_data.edges = []
        self.scale_graph_data.expand_intervals = False
        try:
            self.scale_graph_data.maxcluster = cutoff.maxcluster
        except AttributeError:
            self.scale_graph_data.maxcluster = None

    def nodes_from_path(self, filt):
        '''
        Apply a path through the dendrograms to a Mapper output.
        '''
        if filt.ndim == 1:
            filt = filt[:,np.newaxis]
        assert filt.ndim == 2

        num_nodes_path = self.scale_graph_data.path
        self.clear_nodes()
        for i, (dataidx, Z, diam), num_clust in zip(count(),
                                                     self.scale_graph_data,
                                                     num_nodes_path):
            if num_clust==0:
                points_clusters = np.zeros(dataidx.size,dtype=np.int)
            else:
                points_clusters = fcluster(Z, num_clust)
                #assert num_clust == points_clusters.max()
                assert np.all(np.unique(points_clusters)==\
                                  np.arange(num_clust)), \
                                  ('Number of clusters is not as expected: '
                                   '{0} instead of {1}. Number of points '
                                   'available: {2}.'.\
                                       format(points_clusters.max()+1,
                                              num_clust, dataidx.size))
            # Special case: num_clust==0 can either indicate an empty
            # levelset or a scale above the diameter of a non-empty
            # levelset.
            if num_clust==0 and points_clusters.size>0:
                num_clust=1
            for cl in range(num_clust):
                points = dataidx[ points_clusters == cl ]
                attribute = np.median(filt[points,0])
                if self.point_labels is not None:
                    points = self.point_labels[ points ]
                self.add_node((i,), points, attribute )

        self.add_nodes_to_levelsets()

    def complex_from_nodes(self, cover=None, verbose=True, simple=False):
        if simple:
            self.generate_graph(cover=cover, verbose=verbose)
        else:
            self.generate_complex(cover=cover, verbose=verbose)
        if verbose:
            print ('Dimension: {0}'.format(self.dimension))
        sizes = self.node_sizes.values()
        self.add_info(size_min=min(sizes),size_max=max(sizes))

    def draw_scale_graph(self, **kwargs):
        return draw_scale_graph(self.scale_graph_data, **kwargs)

    def save_scale_graph_as_pdf(self, *args, **kwargs):
        return save_scale_graph_as_pdf(self.scale_graph_data, *args, **kwargs)

    draw_2D = draw_2D

class levelset:
    '''Levelset data structure'''
    def __init__( self, filter_min, filter_max, nodes ):
        '''
        Create a new levelset object.

        @param filter_min: usually a number or a tuple of numbers
        @type filter_min: not specified
        @param filter_max:
        @type filter_max: not specified
        @param nodes: The indices of all nodes in that level set. The nodes
            will be stored as a set, ie. unordered.
        @type nodes: any iterable of nonnegative integers.
        '''
        if nodes is None:
            nodes = []
        else:
            for n in nodes:
                assert isinstance(n, (int, np.integer))
                assert n >= 0

        self.filter_min = filter_min
        self.filter_max = filter_max
        self.nodes = set(nodes)

    def __repr__(self):
        return 'levelset({0},{1},{2})'\
            .format(repr(self.filter_min),
                    repr(self.filter_max),
                    repr(self.nodes) )

    def __str__( self ):
        return "Levelset object" + \
            "\nRange min: " + str( self.filter_min ) + \
            "\nRange max: " + str( self.filter_max ) + \
            "\nNodes: " + str( self.nodes )


class node:
    '''Node data structure'''
    def __init__( self, level, points, attribute ):
        '''
        Create a new node object.

        @param level: level identifier, usually an integer or a tuple of integers
        @type level: not specified
        @param points: nonnegative integers. These are the indices of the points
        in the data set which are contained in the node.
        @type points: numpy.ndarray(N, dtype=int)
        @param attribute: any attribute, eg. a color
        '''
        assert isinstance(points, ndarray)
        assert points.ndim == 1
        assert points.dtype.kind == 'i', \
            'Expected integer data type but got ' + points.dtype
        for n in points:
            assert n>=0

        self.level = level
        self.attribute = attribute
        self.points = points
        # Always maintain a sorted list of points
        self.points.sort()

    def __repr__(self):
        return 'node({0},{1},{2})'\
            .format(repr(self.level), repr(self.points), repr(self.attribute) )

    def __str__( self ):
        return "Node object" + \
            "\nLevel: " + str(self.level) + \
            "\nPoint set: " + str( self.points ) + \
            "\nAttribute: " + str(  self.attribute )

class simpl_complex:
    '''
    Simplicial complex data structure

    The internal data representation is a list of dictionaries. The ith item
    C{self.simplices[i]} is the dict of i-dimensional simplices.
    The keys of C{self.simplices[i]} are (i+1)-tuples of integers (the vertices),
    and each value is a weight for the simplex.

    A simplex is represented as a tuple of integers. Each integer represents
    a vertex. Eg (0,5) represents the 1-simplex on the 0th and 5th vertices.

    @note: This data structure treats simplices as B{ordered} simplices. Eg. the
    simplex (1,3,0) is different from the simplex (3,0,1).
    '''

    def __init__( self, data={} ):
        '''
        Generate a simplicial complex

        @param data: A dict of simplices and weights. Or an iterable of simplices.
                     A C{dionysus.Filtration} is a possible input.
        @type data: dictionary or iterable

        For input purposes a simplex object can be a tuple of integers representing
        vertices( our internal representation ), or a Dionysus Simplex object.
        The latter will be converted into the internal representation.

        When input as a dictionary, the keys are the simplex objects and the values
        are weights. When input as a list, the items are the simplex objects;
        in this case the weights default to 1.

        The simplex objects need not be all of the same dimension. They will be
        sorted into the right postion.
        '''
        self.simplices = []
        if isinstance(data, dict):
            for simplex, weight in dict_items(data):
                self.add_simplex(simplex, weight)
        else:
            for simplex in data:
                self.add_simplex(simplex)

    def __str__(self):
        if len(self.simplices) == 0:
            ret = 'Empty simplicial complex'
        else:
            ret = 'Simplicial complex with the following simplices:'
            for i, simplices in enumerate(self.simplices):
                ret += '\n  Dimension {0}: {1}'.format(i,str(simplices))
        return ret

    def __repr__(self):
        str = 'simpl_complex(' + repr(self.as_dict()) + ')'
        return str

    def __getitem__( self, key ):
        '''
        Direct access to the simplices of a given dimension
        '''
        return self.simplices[key]

    def __setitem__( self, key, value):
        '''
        Direct access to the simplices of a given dimension
        '''
        # Extend the list if necessary to new dimensions
        # Mostly a no-op
        for j in range(len(self.simplices), key+1):
            self.simplices.append(dict())
        self.simplices[key] = value

    def as_dict(self):
        '''
        Represent the data by a dictionary. All simplices are merged into one
        dictionary, regardless of the dimension. Note that the dimension of a
        simplex can always be reconstructed by the key length of the dictionary
        entry.

        @rtype: dictionary
        '''
        simpl_dict = {}
        for simplices in self.simplices:
            simpl_dict.update(simplices)
        return simpl_dict

    def add_simplex(self, vertices, weight=1):
        '''
        Add a simplex to the simplicial complex. Specify the simplex by the
        tuple of its vertices. The procedure takes care to sort it into
        the right dimension.

        Eg: C{add_simplex((0,5), 3)} adds an edge between the vertices
        no. 0 and 5 with weight 3.

        Alternatively, you may use the syntax C{add_simplex(Simplex)}, where C{Simplex} is a
        C{dionysus.Simplex} object. In this case, the I{weight} argument must not be present, and
        the weight is taken from the attribute in the Simplex data structure.

        @attention: A Dionysus Simplex object always returns its simplices in sorted order. This
        destroys the orientation of a simplex, if you are working with oriented or even
        ordered simplices.

        @param vertices: a simplex, given by its vertices
        @type vertices: iterable with nonnegative integers
        @param weight: the weight of the simplex (default 1)
        '''
        #try:
        #    dionysus_simplex = isinstance(vertices, Simplex)
        #except NameError:
        #    dionysus_simplex = False
        #if dionysus_simplex:
        #    assert weight==1
        #    weight = vertices.data
        #    vertices = vertices.vertices
        vertices = tuple(vertices)
        assert len(vertices) >= 1
        for x in vertices:
            assert isinstance(x, (int, np.integer))
            assert x >= 0
        i = len(vertices) - 1# index 0 for vertices
        # Extend the list if necessary to new dimensions
        # Mostly a no-op
        for j in range(len(self.simplices), i+1):
            self.simplices.append(dict())
        self.simplices[i][vertices] = weight

    @property
    def dimension(self):
        '''
        The dimension of the simplicial complex.
        −1 means that the complex is empty.

        Note that the dimension may increase when simplices are added.

        @return: dimension
        @rtype: integer S{>=}−1
        '''
        return len(self.simplices) - 1

    def remove_all_faces(self):
        '''
        Remove all non-maximal simplices down to the triangles (but keep the
        1-skeleton).

        This is a helper method for drawing a simplicial complex. Note that
        the resulting data structure is incomplete: it is not a simplicial
        complex any more. However, the method generates a minimal
        representation for drawing.
        '''
        for dim in range(self.dimension, 2, -1):
            for simplex in self[dim]:
                self.__remove_faces(simplex)

    def __remove_faces(self, simplex):
        '''
        Remove all faces of a given simplex down to the triangles (but keep
        the 1-skeleton).

        @param simplex: Simplex
        @type simplex: ordered list of integers
        '''
        dim = len(simplex)-1
        for f in combinations(simplex, dim): # iterator over all the (codim 1)-faces!
            if f in self[dim-1]:
                del self[dim-1][f]
                if dim>3:
                    self.__remove_faces(f)

    def boundary(self, sanitize=True):
        '''
        Simple method for the B{unoriented} boundary of a simplicial complex.
        This gives the mod-2 boundary of the top-dimensional cells in the
        simplicial complex.

        A good boundary method for chains would be more appropriate, hence the
        present method should not be the last word.

        @param sanitize: Sanitize the result by adding lower-dimensional faces?
        @type sanitize: bool

        @return: boundary
        @rtype: L{simpl_complex}
        '''
        # First step: boundary of all top-dimensional simplices as a chain
        B = defaultdict(int)
        for s in self[-1]:
            for f in combinations(sorted(s), self.dimension):
                B[f] += 1
        # Second step: return all faces with odd coefficients
        S = simpl_complex([s for s, num in dict_items(B) if num%2==1])
        if sanitize:
            S.sanitize_faces()
        return S

    def sanitize_faces(self):
        '''
        Add missing faces to the simplicial complex.
        '''
        for dim in xrange(self.dimension, 0, -1):
            for s in self[dim]:
                for f in combinations(s, dim):
                    if not self[dim-1].has_key(f):
                        self.add_simplex(f)

    def to_graph_tool_Graph(self):
        '''
        Convert the 1-skeleton of a simplicial complex  to a graph_tool Graph.
        The nodes are nonnegative integers.
        No C{info} or C{levelset} dictionary, just the graph itself.
        @rtype: C{graph_tool.Graph}
        '''
        import graph_tool as gt

        vertices = self.vertices()
        G = gt.Graph(directed=False)
        # inverse table
        if vertices.size:
            node_to_vertex = np.empty(vertices[-1]+1, dtype=np.int)
            node_to_vertex[vertices] = np.arange(vertices.size)

            G.add_vertex(vertices.size)
            if self.dimension>=1:
                for v0, v1 in self[1]:
                    G.add_edge(G.vertex(node_to_vertex[v0]),
                               G.vertex(node_to_vertex[v1]))
        return G, vertices

    def remove_small_simplices(self, minsizes):
        if not minsizes:
            return self
        S = simpl_complex()
        # vertices
        minsize = minsizes[0]
        vertices = dict([(k,v) for k,v in dict_items(self[0]) if v>=minsize])
        if vertices:
            S.simplices = [vertices]
        else:
            return S # empty simplicial complex
        vertex_present = np.zeros(self.vertices()[-1]+1, dtype=np.bool)
        vertex_present[S.vertices()] = True
        # higher simplices
        for dim in range(1,self.dimension+1):
            minsize = 1 if len(minsizes)<=dim else minsizes[dim]
            d = dict()
            for k,v in dict_items(self[dim]):
                if v>=minsize and np.all(vertex_present[list(k)]):
                    d[k] = v
            if d:
                S.simplices.append(d)
            else:
                break
        return S

    def vertices(self):
        '''
        Return a the list of vertices. Simple list, no attributes, in sorted
        order.
        '''
        if self.dimension>=0:
            return np.sort(np.array([x for x, in self[0]], dtype=np.int))
        else:
            return np.empty(0, dtype=np.int)

class scale_graph_data:
    '''
    Data structure for the scale graph algorithm.

      - dataidx: list of indices to the data points in the filter patch
      - dendrogram: the dendrogram (C{fastcluster.linkage} output) of the partial
        clustering
      - diam: the diameter of the current data range
    '''
    def __init__(self):
        self.dataidx = []
        self.dendrogram = []
        self.diameter = []

    def reserve(self, num):
        assert not self.dataidx
        assert not self.dendrogram
        assert not self.diameter
        self.dataidx = [None] * num
        self.dendrogram = [None] * num
        self.diameter = [None] * num

    def append(self, dataidx, dendrogram, diameter, levelindex):
        assert len(levelindex)==1
        index = levelindex[0]
        assert index<len(self.dataidx)
        self.dataidx[index] = dataidx
        self.dendrogram[index] = dendrogram
        self.diameter[index] = diameter

    def __len__(self):
        assert len(self.dataidx)==len(self.dendrogram)==len(self.diameter)
        return len(self.dataidx)

    def __getitem__(self, i):
        return (self.dataidx[i], self.dendrogram[i], self.diameter[i])

    def __str__(self):
        return 'Scale graph data for {0} levels.'.format(len(self.dataidx))

    def node_size(self, i):
        if i>=self.N:
            return self.H[i-self.N,3].astype(np.int)
        else:
            return 1

    def set_yaxis(sgd, ax, log):
        if log:
            ylim = sgd.log_limits
            if ylim[0]>0:
                ax.set_yscale('log', nonposy='clip')
            else:
                ax.set_yscale('symlog', linthreshy=-10*ylim[0])
        else:
            ylim = sgd.lin_limits
            ax.set_yscale('linear')
        ax.set_ylim(ylim)
        return ylim

    def layerdata(self, layer):
        '''For the scale graph algorithm.'''
        self.H = self.dendrogram[layer]
        diam = self.diameter[layer]
        if self.H is None:
            self.N = 0
            LB = np.zeros(1)
            UB = LB
        else:
            self.N = np.alen(self.H)+1

            nodesizes = [(self.node_size(j), self.node_size(k)) for j, k in \
                             self.H[::-1,:2].astype(np.int)]

            if self.expand_intervals:
                coeff = [float((a-b)*(a-b))/(a*b*(a+b))
                         for a,b in nodesizes]

            self.H = np.hstack((diam, self.H[::-1,2], 0.))
            if np.any(np.diff(self.H)>0.):
                raise AssertionError('The scale graph algorithm works only '
                                     'for dendrograms without inversions. '
                                     'Also, the largest merging distance '
                                     'must not be larger than the diameter of '
                                     'the singleton set.')

            if self.expand_intervals:
                UB = self.H.copy()
                for j in range(1, len(UB)-1):
                    UB[j] += (UB[j-1]-UB[j])*coeff[j-1]

                LB = self.H.copy()
                for j in range(len(LB)-2,0,-1):
                    LB[j] += (LB[j+1]-LB[j])*coeff[j-1]
            else:
                UB = self.H
                LB = self.H

            #for i in range(len(self.H)):
            #    assert self.H[i]<=UB[i], (i, self.N-1, self.H[i], UB[i])
            #    assert LB[i]<=self.H[i], (i, self.N-1, LB[i], self.H[i])

            del self.H

        return self.N, LB, UB, diam

    draw_scale_graph = draw_scale_graph

def fcluster(Z, num_clust):
    '''Generate cluster assignments from the dendrogram Z. The parameter
    num_clust specifies the exact number of clusters. (The method in SciPy
    does not always produce the exact number of clusters, if several heights
    in the dendrogram are equal, or if singleton clustering is requested.)

    This method starts labeling clusters at 0 while the SciPy indices
    are 1-based.'''
    assert isinstance(num_clust, (int, np.integer))
    N = np.alen(Z)+1
    assert 0<num_clust<=N

    if num_clust==1: # shortcut
        return np.zeros(N, dtype=np.int)

    Z = Z[:N-num_clust,:2].astype(np.int)

    # Union-find data structure
    parent = np.empty(2*N-num_clust, dtype=np.int)
    parent.fill(-1)

    for i,(a,b) in enumerate(Z):
        parent[a] = parent[b] = N+i

    clust = np.empty(N, dtype=np.int)
    for i in range(N):
        idx = i
        if (parent[idx] != -1 ): # a → b
            p = idx
            idx = parent[idx]
            if (parent[idx] != -1 ): # a → b → c
                while True:
                    idx = parent[idx];
                    if parent[idx]==-1: break
                while True:
                    parent[p], p = idx, parent[p]
                    if parent[p]==idx: break
        clust[i] = idx

    # clust contains cluster assignments, but not necessarily numbered
    # 0...num_clust-1. Relabel the clusters.
    idx = np.unique(clust)
    idx2 = np.empty_like(parent)
    idx2[idx] = np.arange(idx.size)

    return idx2[clust]

# Load the C++ routines, if available.
try:
    from cmappertools import fcluster
except ImportError:
    sys.stderr.write("The 'cmappertools' module could not be imported.\n")
