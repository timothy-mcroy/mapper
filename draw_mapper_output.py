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
'''
Drawing routines for Mapper output.
'''
import numpy as np
import matplotlib as mpl
try:
    import matplotlib.patheffects as patheffects
    patheffects_available = True
except ImportError:
    print("Warning: The module matplotlib.patheffects cannot be imported.")
    patheffects_available = False
import sys
import re
from math import ceil, floor, log
from operator import getitem
from itertools import combinations, count
if sys.hexversion < 0x03000000:
    from itertools import izip as zip
    range = xrange
from mapper.tools import qhull, shortest_path

import networkx as nx
# More imports in the 3D section

mplversion = [int(x) for x in mpl.__version__.split(' ')[0].split('.')[:2]]
if mplversion < [1, 0]:
    print ('Warning: Best results are obtained with Matplotlib version at '
           'least 1.0.\n'
           'Python Mapper should work with Matplotlib 0.99, too, but some '
           'features are\nmissing.')
if mplversion < [1, 1]:
    # Specification for a short horizontal line, 'hline'.
    # We cannot use the first specification universally since Matplotlib
    # changed the angle parameter from radians to degree in 1.1.0.
    hlinemarkersymbol = (2, 2, .5 * np.pi)
else:
    # (2,2,90) does work, too, but Matplotlib 1.1.0 knows the symbol '_'.
    hlinemarkersymbol = '_'

__all__ = ['draw_2D', 'draw_3D', 'draw_scale_graph', 'save_scale_graph_as_svg',
           'scale_graph_axes_args', 'scale_graph_axes_kwargs']

# TBD:Rewrite draw_2D so that the output is an object that has efficient
# self.update_node_labels, .update_node_colors, .update_node_boundary, .update_node_boundaries etc methods
#
# This will simplify and increase efficiency of the GUI's MapperOutputFrame.

def floats_to_rgbas(colors_as_floats, color_range_min, color_range_max):
    # Takes in an iterable of floats and returns an iterable of rgb tuples
    # Deals with non finite floats by setting them to black

    black_list = []
    for index, c in enumerate(colors_as_floats):
        assert isinstance(c, (float, np.floating))
        if not np.isfinite(c):  # Eg nan, Inf etc
            black_list.append(index)
            # for now set the special values to color_range_min
            # later on we'll black it out
            colors_as_floats[index] = color_range_min

    # get current color map
    cmap = mpl.cm.get_cmap()
    norm = mpl.colors.Normalize(vmin=color_range_min, vmax=color_range_max)
    cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors_as_rgbas = cm.to_rgba(colors_as_floats)

    colors_as_rgbas[black_list] = (0, 0, 0, 1)
    return colors_as_rgbas

def draw_2D(M, ax=None, node_labels=None, node_colors=None, legend=True, verbose=False, node_scale=1000., rotate=True, minsizes=(), node_labels_scheme=None, node_colors_scheme=None):
    '''
    Draw a Mapper output in 2D. The Mapper output may have higher-dimensional simplices.

    Characteristics:
      - Graphviz's 'neato' is used to determine the node positions.
      - The node sizes are related (but not proportional) to the number
        of data points in the node. Moreover each node is labeled with
        the number of data points in the node.
      - The nodes colors correspond to the node attribute (it must be a float).
        It is assumed that each node attribute refers to a filter value.
        Therefore, the full color range is taken from the full filter range.
      - Projections of higher-dimensional simplices are drawn into the 2D figure.
      - The edge thickness is related to the number of points in the intersection of two nodes.

    @todo: Decide what to write into the legend.
    @todo: Transparency of the high-dimensional simplices in relation to the number of points in the
    intersection?
    @todo: Rename this module as soon as the name L{draw_mapper_output} becomes free.

    @param M: Mapper output
    @type M: L{mapper_output}
    @param verbose: print progress messages?
    @type verbose: bool
    @param rotate: rotate the diagram so that its largest extent is horizontal
    @type rotate: bool
    '''
    if ax:
        fig = ax.get_figure()
    else:
        # draw the figure via matplotlib
        # from matplotlib.figure import Figure
        # fig = Figure(facecolor='w')
        from matplotlib.pyplot import figure
        fig = figure(facecolor='w')
        # full window, no coordinate axes
        ax = fig.add_axes((0, 0, 1, 1), aspect='equal')
    ax.set_axis_off()
    # ax.set_clip_on(True)
    # ax.set_clip_path(None)
    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax.set_frame_on(False)
    ax.set_autoscale_on(False)

    S = M.simplices
    if minsizes:
        S = S.remove_small_simplices(minsizes)
    # Caution: Not all nodes may be vertices!
    G, vertices = S.to_nx_Graph()

    # tbd: 'size' and 'ratio' parameter
    vertex_pos = nx.graphviz_layout(G, prog='neato')
    vertex_pos = np.array([vertex_pos[v] for v in vertices])

    if not vertex_pos.size:
        vertex_pos = np.empty((0, 2))
        x = np.empty((0, 2))
        y = np.empty((0, 2))
        minx = miny = maxx = maxy = 0.
    elif rotate:
        # Generate the enclosing polygon of the nodes
        # Find the smallest area enclosing rectangle with the rotating
        # caliper method.
        ch = qhull(vertex_pos)  # already a polygon with vertices in a cycle
        phi, MP = enclosing_rect(ch)
        minx, maxx, miny, maxy = MP
        # landscape format is preferred over portrait
        if maxy - miny > maxx - minx:
            phi += np.pi * .5
            minx = -MP[3]
            maxx = -MP[2]
            miny = MP[0]
            maxy = MP[1]

        vertex_pos = np.dot(vertex_pos, ((np.cos(phi), np.sin(phi)),
                                         (-np.sin(phi), np.cos(phi))))

        # As a convenience, keep the lower left quadrant as free as possible
        # from vertices to make room for the legend.
        left = vertex_pos[:, 0] < .5 * (minx + maxx)
        down = vertex_pos[:, 1] < .5 * (miny + maxy)
        ld = np.logical_and(left, down).sum()  # count_nonzero() n/a in NumPy
        l_count = left.sum()  # <1.6.0
        lu = l_count - ld
        rd = down.sum() - ld
        ru = len(vertex_pos) - l_count - rd
        quadrant = np.argmin((ld, rd, lu, ru))
        if quadrant > 1:  # up
            vertex_pos[:, 1] = -vertex_pos[:, 1]
            miny, maxy = -maxy, -miny
        if quadrant % 2:  # right
            vertex_pos[:, 0] = -vertex_pos[:, 0]
            minx, maxx = -maxx, -minx
        x = vertex_pos[:, 0]
        y = vertex_pos[:, 1]
    else:
        x = vertex_pos[:, 0]
        y = vertex_pos[:, 1]
        minx = min(x)
        miny = min(y)
        maxx = max(x)
        maxy = max(y)

    # table for node positions; some rows may remain empty
    node_pos = np.empty((M.num_nodes, 2))
    node_pos[vertices, :] = vertex_pos

    S.remove_all_faces()

    # Semi-transparent higher simplices (dim>3) in yellow.
    for dim in range(S.dimension, 3, -1):
        if verbose:
            print("Draw the simplices of dimension {0}.".format(dim))
        for poly in S[dim]:
            coor = node_pos[poly, :]
            # Form the convex hull!
            coor = qhull(coor)
            p = mpl.patches.Polygon(coor, edgecolor='none', facecolor='y',
                                    alpha=.8, zorder=0)
            ax.add_patch(p)

    # Semi-transparent tetrahedra in red.
    if S.dimension > 2:
        if verbose:
            print("Draw the tetrahedra.")
        for tetra in S[3]:
            coor = node_pos[tetra, :]
            # Form the convex hull!
            coor = qhull(coor)
            p = mpl.patches.Polygon(coor, edgecolor='none', facecolor='r',
                                    alpha=.6, zorder=0)
            ax.add_patch(p)

    # Semi-transparent triangles in blue.
    if S.dimension > 1:
        if verbose:
            print("Draw the triangles.")
        for triangle in S[2]:
            coor = node_pos[triangle, :]
            p = mpl.patches.Polygon(coor, edgecolor='none', facecolor='b',
                                    alpha=.5, zorder=0)
            ax.add_patch(p)

    # Relate the line thickness to the weight of each edge.
    if S.dimension >= 1:
        edgecoor = node_pos[S[1].keys(), :]
        edge_weights = np.array(S[1].values(), dtype=np.float)
        linewidths = .1 + 3 * (edge_weights / edge_weights.max()) ** .5
        edges = mpl.collections.LineCollection(segments=edgecoor,
                                               linewidths=linewidths,
                                               colors=((0, 0, 0, 1),),
                                               )
        ax.add_collection(edges)

    info = M.info
    if S.dimension < 0:
        circles = mpl.collections.CircleCollection([])
    else:
        if node_labels_scheme is not None:
            info["node_labels_scheme"] = node_labels_scheme
        if node_colors_scheme is not None:
            info["node_colors_scheme"] = node_colors_scheme
        filter_min = info['filter_min']
        filter_max = info['filter_max']

        # Determine sizes and colors
        vertex_sizes = np.array([M.node_sizes[v] for v in vertices],
                                dtype=np.int)
        # Formula for the circle sizes as in Gurjeet Singh's Mapper code
        vertex_area = node_scale * (.1 + vertex_sizes / float(vertex_sizes.max()))

        if node_colors is None:
            node_colors = [M.nodes[v].attribute for v in vertices]
            color_range_min, color_range_max = filter_min, filter_max
        elif hasattr(node_colors, "__call__"):
            node_colors = np.array([node_colors(M.nodes[v]) for v in vertices],
                                   dtype="float")
            color_range_min, color_range_max = \
                min(node_colors), max(node_colors)
        else:
            color_range_min, color_range_max = \
                min(node_colors), max(node_colors)

        # TBD check lengths of node_colors and node_labels are correct
        node_colors_rgba = floats_to_rgbas(node_colors, color_range_min,
                                           color_range_max)

        circles = mpl.collections.CircleCollection(vertex_area,
                                                   offsets=vertex_pos,
                                                   linewidths=(.5,),
                                                   facecolors=node_colors_rgba,
                                                   zorder=2,
                                                   transOffset=ax.transData)

        # from the matplotlib example - not sure why this does
        # trans = fig.dpi_scale_trans + \
        #    mpl.transforms.Affine2D().scale(1./72.)
        # circles.set_transform(trans)  # the points to pixels transform
        ax.add_collection(circles)

        if node_labels is None:
            node_labels = vertex_sizes
        elif hasattr(node_labels, "__call__"):
            node_labels = [ node_labels(M.nodes[v]) for v in vertices ]
        elif node_labels != 'empty':
            node_labels = [node_labels[v] for v in vertices]

        if node_labels != 'empty':
            if patheffects_available:
                textkwargs = dict([('path_effects', \
                                        [patheffects.withStroke(\
                                    linewidth=2,
                                    foreground=(.3, .3, .3))])])
            else:
                textkwargs = dict()
            for xx, yy, label in zip(x, y, node_labels):
                ax.text(xx, yy, label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=.85 * mpl.rcParams['font.size'],
                        color='w',
                        **textkwargs
                        )

    # legend
    # This is subject to change!
    if legend:
        legend = []
        if 'filter_min' in info:
            legend.append('Filter range: [{0:.2f}, {1:.2f}]' \
                              .format(info['filter_min'], info['filter_max']))
        if 'filter_min_array' in info:
            filter_min_array = info['filter_min_array']
            if len(filter_min_array) > 1:
                legend.append('Filter minima: ' + str(filter_min_array))
        if 'filter_max_array' in info:
            filter_max_array = info['filter_max_array']
            if len(filter_max_array) > 1:
                legend.append('Filter maxima: ' + str(filter_max_array))
        if 'cover' in info:
            legend.append('Cover: {0}'.format(info['cover']['str']))
        if 'cluster' in info:
            legend.append('Clustering method: {0}'.format(info['cluster']))
        if 'cutoff' in info:
            legend.append('Cutoff: {0}'.format(info['cutoff']))
        if 'size_min' in info:
            legend.append('Size range: [{0},{1}]' \
                          .format(info['size_min'], info['size_max']))
        if 'node_labels_scheme' in info:
            legend.append('Vertices labelled by ' + \
                               str(info['node_labels_scheme']))
        if 'node_colors_scheme' in info:
            legend.append('Vertices colored by ' + \
                               str(info['node_colors_scheme']))

        legend = '\n'.join(legend)
        ax.text(.01, .01, legend, ha='left', va='bottom',
                 transform=ax.get_figure().transFigure)

    # Heuristic to the 'padding' in the diagram
    # The problem: the nodes have constant pixel size, independent of the
    # magnification, so the exact padding depends on magnification and
    # output size (screen size, paper size).

    # 1.1*node_scale = max(node_sizes)
    # +2: 2 pixels extra margin
    # m: margin in inches
    margin_inches = (np.sqrt(1.1 * node_scale / np.pi) / 72. + 2. / ax.get_figure().dpi)

    denomx = maxx - minx
    size = fig.get_size_inches()
    if denomx < 0:
        raise AssertionError
    elif denomx == 0:
        scalex = np.inf
    else:
        scalex = (size[0] - 2 * margin_inches) / denomx
    denomy = maxy - miny
    if denomy < 0:
        raise AssertionError
    elif denomy == 0:
        scaley = np.inf
    else:
        scaley = (size[1] - 2 * margin_inches) / denomy

    scale = min(scalex, scaley)
    if np.isinf(scale):
        scale = 1
    if scale <= 0: raise AssertionError

    ms = margin_inches / scale
    xlim = (minx - ms, maxx + ms)
    ylim = (miny - ms, maxy + ms)
    if xlim[0] == xlim[1]: xlim = (xlim[0] - .5, xlim[1] + .5)
    if ylim[0] == ylim[1]: ylim = (ylim[0] - .5, ylim[1] + .5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return (minx, miny, maxx, maxy, margin_inches), \
        vertices, vertex_pos, circles

def enclosing_rect(ch):
    '''Find the smallest area enclosing rectangle with the rotating caliper
    method. See http://cgm.cs.mcgill.ca/~orm/maer.html

    The points on the convex hull polygon must be in clockwise order.
    '''
    if not ch.size:
        return 0, (0., 0., 0., 0.,)
    elif np.alen(ch) == 2:  # one point
        return 0, ch[0][[0, 0, 1, 1]]
    assert np.alen(ch) > 2  # from two points on: closed polygon
    assert np.all(ch[-1] == ch[0])  # closed polygon required
    # Make sure that the convex polygon goes around clockwise
    edges = np.diff(ch, axis=0)
    edges_rotated = np.column_stack((edges[:, 1], np.negative(edges[:, 0])))
    assert np.all(np.sum(edges * np.roll(edges_rotated, 1, axis=0), axis=1) >= 0.), \
        'The polygon must be traversed in clockwise order.'
    minimum_area = np.inf
    optimal_angle = None
    bounding_box = None
    total_rotation = 0.
    # bboxindices contains indices to [minx, miny, maxx, maxy]
    bboxindices = np.hstack((ch.argmin(axis=0), ch.argmax(axis=0)))
    # If you transribe this code to a different programming language:
    # I require here that argmin chooses the lowest possible index
    # if the minimum is attained several times (eg. at indices 0, -1).
    for _dummy in ch:
        zero_angles = True
        while np.any(zero_angles):
            # convert index 0 to last index
            bboxindices = (bboxindices - 1) % (np.alen(ch) - 1) + 1
            # four edge vectors at the extremal vertices
            d = ch[bboxindices - 1] - ch[bboxindices]
            # rotate by (90‌°, 0‌°, 270‌°, 180‌°)
            d[0, :] = (-d[0, 1], d[0, 0])
            d[2, :] = (d[2, 1], -d[2, 0])
            d[3, :] = -d[3, :]
            angles = .5 * np.pi - np.arctan2(*d.T)
            assert np.all(angles >= -1e-7)  # tbd: remove
            assert np.all(angles <= np.pi + 1e-7)
            angles = np.minimum(angles, np.pi)  # numerical stability
            zero_angles = angles <= 0.  # Filter out zero angles and go to the
            bboxindices[zero_angles] -= 1  # previous index.

        minidx = angles.argmin()
        rotation_angle = angles[minidx]
        assert rotation_angle > 0.

        ch = np.dot(ch, ((np.cos(rotation_angle), -np.sin(rotation_angle)),
                         (np.sin(rotation_angle), np.cos(rotation_angle))))
        newbboxindices = np.hstack((ch.argmin(axis=0), ch.argmax(axis=0)))
        newbboxindices[minidx] = (bboxindices[minidx] - 1) % (np.alen(ch) - 1)
        bboxindices = newbboxindices

        total_rotation -= rotation_angle
        area = (ch[bboxindices[3], 1] - ch[bboxindices[1], 1]) * \
            (ch[bboxindices[2], 0] - ch[bboxindices[0], 0])
        if area < minimum_area:
            minimum_area = area
            optimal_angle = total_rotation
            bounding_box = ch[bboxindices[[0, 2, 1, 3]], [0, 0, 1, 1]]
        if total_rotation < -.5 * np.pi:
            assert optimal_angle is not None
            return (optimal_angle, bounding_box)

    raise RuntimeError('Algorithm did not converge. ' \
                           'Please file a bug report to Daniel.')

def pygraphviz_layout_3D(G, prog='neato', root=None, args='-Gdim=3 -Nz=""'):
    '''
    Create node positions in 3D for G using Graphviz.

    Derived from C{networkx.pygraphviz_layout}.

    Example
    =======
    >>> G=nx.petersen_graph()
    >>> pos=nx.graphviz_layout(G)
    >>> pos=nx.graphviz_layout(G,prog='dot')

    @param G: A graph created with NetworkX
    @type G: NetworkX graph
    @param prog: Name of Graphviz layout program
    @type prog: string
    @param root: Root node for twopi layout
    @type root: string, optional
    @param args: Extra arguments to Graphviz layout program
    @type args: string, optional

    @rtype: dictionary
    @return: Dictionary of M{(x,y,z)} positions keyed by node.
    '''
    import networkx
    A = networkx.to_agraph(G)
    if root is not None:
        args += "-Groot=%s" % root
    A.layout(prog=prog, args=args)
    node_pos = {}
    for node in A:
        n = int(node.get_name())
        x, y = node.attr["pos"].split(',')
        z = node.attr["z"]
        node_pos[n] = (float(x), float(y), float(z))
    return node_pos

def draw_3D(M, verbose=False):
    '''
    Draw a Mapper output in 3D. The Mapper output may have higher-dimensional simplices.

    Characteristics:
      - Graphviz's 'neato' is used to determine the node positions.
      - The node sizes are related (but not proportional) to the number
        of data points in the node. Moreover each node is labeled with
        the number of data points in the node.
      - The nodes colors correspond to the node attribute (it must be a float).
        It is assumed that each node attribute refers to a filter value.
        Therefore, the full color range is taken from the full filter range.
      - Projections of higher-dimensional simplices are drawn into the 2D figure.
      - The edge thickness is related to the number of points in the intersection of two nodes.

    @todo: Decide what to write into the legend.
    @todo: Transparency of the high-dimensional simplices in relation to the number of points in the
    intersection?
    @todo: Rename this module as soon as the name L{draw_mapper_output} becomes free.

    @param M: Mapper output
    @type M: L{mapper_output}
    @param verbose: print progress messages?
    @type verbose: bool
    '''
    from dionysus import Filtration, fill_alpha3D_complex
    from enthought.mayavi import mlab
    from mapper_output import simpl_complex

    M_iter = range(M.num_nodes)
    G = M.to_simple_Graph()

    # Determine sizes and colors
    size_dict = M.node_sizes;
    max_size = float(max(size_dict.values()))
    # Formula for the nodes sizes as in Gurjeet Singh's Mapper code
    node_sizes = [.1 + size_dict[i] / max_size for i in M_iter]

    # maximal edge weight
    maxweight = float(max(M.simplices[1].values()));

    colors = [node.attribute for node in M.nodes]
    assert all([isinstance(color, (float, np.floating))  for color in colors])
    mincol = min(colors)
    maxcol = max(colors)
    print(mincol)
    print(maxcol)
    # colors = [color-mincol for color in colors]

    info = M.info
    filter_min = info['filter_min']
    filter_max = info['filter_max']

    # draw the figure via matplotlib
    # fig = plt.figure()
    # fig.set_facecolor('w')
    # full screen, no coordinate axes
    # ax = mpl.axes3d.Axes3D(fig)
    # ax = fig.add_axes((0,0,1,1), aspect='equal')
    # ax.set_axis_off()

    mlab.figure(1, bgcolor=(1, 1, 1))
    # mlab.options.backend = 'envisage'
    mlab.clf()

    pos = pygraphviz_layout_3D(G)

    S = M.simplices
    S.remove_all_faces()

    x = np.array([pos[i][0] for i in M_iter])
    y = np.array([pos[i][1] for i in M_iter])
    z = np.array([pos[i][2] for i in M_iter])
    start_idx = np.array([e[0] for e in S[1].keys()])
    end_idx = np.array([e[1] for e in S[1].keys()])
    # get current cmap
    cmap = mpl.cm.get_cmap()
    norm = mpl.colors.Normalize(vmin=filter_min, vmax=filter_max)
    cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    c = cm.to_rgba(colors)
    # Neglect the transparency values


    zero = np.zeros(np.array(node_sizes).shape)

    # See http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html#changing-the-scale-and-position-of-objects
    pts = mlab.quiver3d(x, y, z,
                  # colors, colors, colors,
                  # scalars = node_sizes,
                  node_sizes, node_sizes, node_sizes,
                  scalars=colors,
                  mode='sphere',
                  scale_factor=20,
                  vmin=filter_min,
                  vmax=filter_max,
                  )
    pts.glyph.color_mode = 'color_by_scalar'
    pts.glyph.scale_mode = 'scale_by_vector'
    # Center the glyphs on the data point
    pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]

    mlab.quiver3d(x[start_idx],
                  y[start_idx],
                  z[start_idx],
                  x[end_idx] - x[start_idx],
                  y[end_idx] - y[start_idx],
                  z[end_idx] - z[start_idx],
                  line_width=2,
                  mode='2ddash',
                  scale_factor=1,
                  color=(0, 0, 0))

    if S.dimension > 1:
        if verbose:
            print("Draw the triangles.")
        idx = np.empty((0, 3), dtype=int)
        for triangle in S[2]:
            idx = np.vstack((idx, triangle))  # tbd
        if idx.size > 0:
            mlab.triangular_mesh(x, y, z, idx,
                color=(0, 0, 1),
                opacity=.3)

    if S.dimension > 2:
        if verbose:
            print("Draw the tetrahedra.")
        idx = np.empty((0, 3), dtype=int)
        for tetra in S[3]:
            for triangle in combinations(tetra, 3):
                idx = np.vstack((idx, triangle))  # tbd
        if idx.size > 0:
            mlab.triangular_mesh(x, y, z, idx,
                color=(1, 1, 0),
                opacity=.3)

    if S.dimension > 3:
        if verbose:
            print("Draw the higher simplices.")
        idx = np.empty((0, 3), dtype=int)
        for dim in range(4, S.dimension + 1):
            for simplex in S[dim]:
                # Generate the 3D Delaunay triangulation
                F = Filtration()
                # Unfortunately, Dionysus does not accept numpy arrays, so we must
                # provide stacked lists.
                coor = [list(pos[i]) for i in simplex]
                fill_alpha3D_complex(coor, F)
                s = simpl_complex(F)
                b = s.boundary(sanitize=False)
                for triangle in b[2]:
                    triangle_idx = [simplex[i] for i in triangle]
                    idx = np.vstack((idx, triangle_idx))  # tbd
        if idx.size > 0:
            mlab.triangular_mesh(x, y, z, idx,
                color=(1, 0, 0),
                opacity=.3)

    '''
    # legend
    # This is subject to change!
    legend = []
    if 'code' in info:
        legend.append('Code: {0}'.format(info['code']))
    if 'filter_min' in info:
        legend.append('Filter range: [{0:.2f}, {1:.2f}]'.format(info['filter_min'],info['filter_max']))
    if 'intervals' in info:
        legend.append('Intervals: {0}'.format(info['intervals']))
    if 'overlap' in info:
        legend.append('Overlap: {0}%'.format(info['overlap']))
    if 'size_min' in info:
        legend.append('Size range: [{0}, {1}]'.format(info['size_min'],info['size_max']))
    if 'gap' in info:
        legend.append('Gap fraction: {0:.2f}'.format(info['gap']))

    legend = '\n'.join(legend)
    plt.figtext(.02,.02,legend,ha='left', va='bottom')
    '''

    # mlab.view(60, 46, 4)
    mlab.show()

scale_graph_axes_args = ((0.06, 0.04, 0.92, 0.94),)
scale_graph_axes_kwargs = {'frameon' : True}

def draw_scale_graph(sgd, ax=None, log=False, maxvertices=None,
                     infinite_edges=None, verbose=True):
    '''
    infinite_edges: True: draw edges with infinite weight
                    False: don't draw
                    None: auto mode, depends on whether the path
                          has infinite weight edges
    '''
    if maxvertices is None:
        maxvertices = sgd.maxcluster
    if sgd.edges and infinite_edges is None:
        infinite_edges = (sgd.infmin > 0)
    maxdiameter = max(sgd.diameter)
    zero_vertex_offset = .05 * maxdiameter
    pathlength = len(sgd.path)
    assert pathlength == len(sgd.dendrogram) == len(sgd.diameter)
    if maxvertices is None:
        totalvertices = sum([len(Z) for Z in sgd.dendrogram \
                                if Z is not None]) + pathlength
    else:
        totalvertices = pathlength * maxvertices

    if ax is None:
        # from matplotlib.figure import Figure
        # fig = Figure(facecolor='w')
        from matplotlib.pyplot import figure
        fig = figure(facecolor='w')
        ax = fig.add_axes(*scale_graph_axes_args,
                           **scale_graph_axes_kwargs)

    # Draw the path
    c = mpl.collections.PolyCollection(_path_bars(sgd), alpha=.1,
                                       edgecolor='none', linewidth=0)
    ax.add_collection(c, autolim=False)

    # Draw the nodes and the bars for the dendrograms
    if verbose:
        print('Draw nodes and dendrograms.')
    node_y = []
    heights_ext = []
    vertlines = np.empty((pathlength, 2, 2))
    marker_x = np.empty(totalvertices + pathlength)
    marker_y = np.empty_like(marker_x)
    marker_i = 0
    dot_x = np.empty(totalvertices)
    dot_y = np.empty_like(dot_x)
    dot_i = 0
    zerodot_y = np.empty(pathlength)
    graylines = np.empty((pathlength, 2, 2))
    graylines_i = 0
    for i, dendrogram, diameter in zip(count(), sgd.dendrogram,
                                       sgd.diameter):
        if diameter is None:
            heights_ext.append(None)
            vertlines[i] = (i, 0.)
            ny = np.array([zero_vertex_offset])
        else:
            y = np.hstack((diameter, dendrogram[::-1, 2], 0))
            heights_ext.append(y)
            vy = .5 * (y[1:] + y[:-1])
            if maxvertices is None:
                y_m = y
                vy_m = vy
            else:
                y_m = y[:maxvertices + 1]
                vy_m = vy[:maxvertices]
                graylines[graylines_i] = ((i, 0.), (i, y_m[-1]))
                graylines_i += 1
            vertlines[i] = ((i, y_m[0]), (i, y_m[-1]))
            marker_x[marker_i:marker_i + len(y_m)].fill(i)
            marker_y[marker_i:marker_i + len(y_m)] = y_m
            marker_i += len(y_m)
            dot_x[dot_i:dot_i + len(vy_m)].fill(i)
            dot_y[dot_i:dot_i + len(vy_m)] = vy_m
            dot_i += len(vy_m)
            ny = np.hstack((diameter + zero_vertex_offset, vy))
        node_y.append(ny)
        zerodot_y[i] = ny[0]
    ax.add_collection(mpl.collections.LineCollection(\
            graylines[:graylines_i], colors='#303030', linewidth=6.),
                      autolim=False)
    ax.add_collection(mpl.collections.LineCollection(vertlines, colors='k'),
                      autolim=False)
    ax.scatter(marker_x[:marker_i], marker_y[:marker_i], c='k',
               marker=hlinemarkersymbol, s=25)
    ax.scatter(dot_x[:dot_i], dot_y[:dot_i], c='r', edgecolor='none',
               linewidth=0,)
    if infinite_edges and sgd.edges:
        ax.scatter(np.arange(pathlength), zerodot_y, c='#0050FF',
                   edgecolor='none', linewidth=0,)

    # Draw the edges
    if verbose and sgd.edges:
        print('Fill edge array.')

    NE = sum(map(len, sgd.edges))
    L = np.empty((NE, 2, 2))
    C = np.empty(NE, dtype=np.unicode_)
    j = 0
    for i, E in enumerate(sgd.edges):
        EE = np.array(E)  # columns: source, target, infinite
        if not infinite_edges:
            EE = EE[EE[:, 2] == 0]
        n = np.alen(EE)
        y1 = node_y[i]
        y2 = node_y[i + 1]
        L[j:j + n, :, 0] = (i, i + 1)
        L[j:j + n, 0, 1] = y1[EE[:, 0]]
        L[j:j + n, 1, 1] = y2[EE[:, 1]]
        C[j:j + n] = np.where(EE[:, 2], 'r', 'k')
        j += n

    lines = mpl.collections.LineCollection(L[:j], linewidths=.2, colors=C[:j])
    ax.add_collection(lines, autolim=False)

    '''
    yseq = np.fromiter(map(getitem, node_y, sgd.path), dtype=float)
    ax.plot(range(yseq.size), yseq, '#00C000', linewidth=1.5)

    SP = shortest_scale_path(sgd)
    #ax.plot(np.linspace(0,yseq.size-1,SP.size), SP, '#40FFFF')
    ax.plot(np.linspace(0,yseq.size-1,SP.size), SP, '#FF8000')
    '''
    SP = shortest_scale_path(sgd)
    ax.plot(np.linspace(0, len(sgd.path) - 1, SP.size), SP, '#00C000',
            linewidth=1.5)

    xlim = (-1, pathlength)
    ax.set_xlim(xlim)
    sgd.lin_limits = (maxdiameter * -.05, maxdiameter * 1.03 + zero_vertex_offset)

    ymax = maxdiameter * 1.03 + zero_vertex_offset
    SPmin = min(SP)
    dotmin = min(dot_y[:dot_i])
    ymin = max(SPmin * SPmin / ymax, dotmin * np.power(dotmin / ymax, .02))
    assert ymin >= 0.
    if ymin > 0:
        ymax *= np.power(ymax / ymin, .02)
    else:
        ymax *= 1.2589254117941673  # np.power(1e5,.02)
        ymin = ymax * -1e-6
    sgd.log_limits = (ymin, ymax)

    ylim = sgd.set_yaxis(ax, log)

    if verbose:
        print('Done.')

    # Return the bounding box (plus None for no node size).
    return (xlim[0], ylim[0], xlim[1], ylim[1], None)

def find_good_stepsize_for_axis_ticks(yrange):

    m, e = re.search("([0-9.]*)e(.*)", "{:e}".format(yrange)).group(1, 2)
    m = float(m) * .1
    e = int(e) + 1
    mult = 1
    while m < 10:
        e -= 1
        mult = 5
        m *= 2
        if m < 10:
            mult = 2
            m *= 2.5
        else:
            break
        if m < 10:
            mult = 1
            m *= 2
        else:
            break

    return pow(10, e) * mult

def find_good_basis_for_axis_ticks(y, dy):
    return ceil(y / dy) * dy

def save_scale_graph_as_svg(sgd, filename, log_yaxis=False, maxvertices=None,
                     infinite_edges=None, verbose=True, bbox=[None, None, None, None]):
    '''
    infinite_edges: True: draw edges with infinite weight
                    False: don't draw
                    None: auto mode, depends on whether the path
                          has infinite weight edges
    '''
    # todo log axis
    if not filename.endswith('.svg'):
        filename += '.svg'
    Width = 201.968503113
    Height = Width * .75
    Linewidth = .4
    Tickwidth = 1
    Fontsize = 5
    Bottomlabeloffset = Fontsize
    Bottomoffset = Bottomlabeloffset
    Topoffset = .5 * Linewidth
    # Font = mpl.rcParams['font.sans-serif'][0]
    Font = 'LMRoman5'

    Teststring = '00000000000000000000'
    Teststring = ''

    maxdiameter = max(sgd.diameter)
    pathlength = len(sgd.path)

    # bbox: left, bottom, right, top
    bbox = list(bbox)
    if bbox[0] is None:
        bbox[0] = -1
    if bbox[1] is None:
        bbox[1] = -.05 * maxdiameter
    if bbox[2] is None:
        bbox[2] = pathlength
    if bbox[3] is None:
        bbox[3] = 1.1 * maxdiameter
        
    W = float(bbox[2] - bbox[0])
    assert W > 0, 'width must be positive'
    assert not log_yaxis or bbox[1] > 0, 'lower bound must be positive for log axis'
    if log_yaxis:
        H = log(bbox[3]) - log(bbox[1])  
    else:
        H = float(bbox[3] - bbox[1])
    assert H > 0, 'height must be positive'
   
    if log_yaxis:
        e0 = int(floor(log(bbox[1], 10)))
        e1 = int(floor(log(bbox[3], 10)))

        maxchars = len(Teststring + "0e{}".format(e0))
        maxchars = max(maxchars, len(Teststring + "0e{}".format(e1)))

    else:
        dy = find_good_stepsize_for_axis_ticks(H)
        sy = find_good_basis_for_axis_ticks(bbox[1], dy)

        maxchars = 0
        for y in np.arange(sy, bbox[3], dy):
            if y == 0:
                y = 0
            maxchars = max(maxchars, len(Teststring + "{:.2g}".format(y)))

    # Factor .64 depends on the font
    Leftoffset = (.64 * maxchars) * Fontsize
    Rightoffset = .5 * Linewidth

    if maxvertices is None:
        maxvertices = sgd.maxcluster
    if sgd.edges and infinite_edges is None:
        infinite_edges = (sgd.infmin > 0)

    def xtransform(x):
        return (x - bbox[0]) / W * (Width - Leftoffset - Rightoffset) + Leftoffset
    
    if log_yaxis:
        def ytransform(y):
            return Height - (log(y) - log(bbox[1])) / H * (Height - Bottomoffset - Topoffset) - Bottomoffset
    else:
        def ytransform(y):
            return Height - (y - bbox[1]) / H * (Height - Bottomoffset - Topoffset) - Bottomoffset

    zero_vertex_offset = .05 * maxdiameter
    assert pathlength == len(sgd.dendrogram) == len(sgd.diameter)
    if maxvertices is None:  # TODO
        maxnumvertices = sum([len(Z) for Z in sgd.dendrogram \
                                if Z is not None]) + pathlength
    else:
        maxnumvertices = pathlength * maxvertices

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
                '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
                '<!-- Created with Python Mapper (http://www.danifold.net/mapper) -->\n'
                '<svg version="1.1" viewBox="0 0 {0} {1}" width="{0}pt" height="{1}pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
                '<g fill="none" stroke-width="{2}" stroke-linecap="butt">\n'.
                    format(Width, Height, Linewidth))
        
        f.write('<defs>\n')
        f.write('<rect id="r" x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" stroke="black" />\n'.
                    format(min(xtransform(bbox[0]), xtransform(bbox[2])),
                           min(ytransform(bbox[1]), ytransform(bbox[3])),
                           abs(xtransform(bbox[0]) - xtransform(bbox[2])),
                           abs(ytransform(bbox[1]) - ytransform(bbox[3]))))
        f.write('</defs>\n')
        f.write('<clipPath id="c">\n')
        f.write('<use xlink:href="#r"/>\n')
        f.write('</clipPath>\n')
        
        xrange = min(pathlength - 1, bbox[2]) - max(0, bbox[0])
        if xrange > 25:
            dx = 5
        elif xrange > 15:
            dx = 2
        else:
            dx = 1

        f.write('<g font-family="{}" font-size="{}" fill="black">\n'.format(Font, Fontsize))
        f.write('<g text-anchor="middle">\n'.format(Fontsize))
        for x in range(max(0, int(ceil(bbox[0]))), min(pathlength - 1, int(floor(bbox[2]))), dx):
            f.write('<text x="{:.2f}" y="{:.2f}">{}</text>\n'.
            format(xtransform(x), Height - Bottomoffset + Bottomlabeloffset, x))
        f.write('</g>\n')

        if log_yaxis:
            for e in range(e0, e1+1):
                y = 10**e;
                if y<=bbox[1] or y>=bbox[3]: continue
                f.write('<text x="{:.2f}" y="{:.2f}">{}1e{}</text>\n'.
                        format(0, ytransform(y) + .37 * Fontsize, Teststring, e))
        else:
            for y in np.arange(sy, bbox[3], dy):
                if y == 0:
                    y = 0
                f.write('<text x="{:.2f}" y="{:.2f}">{}{:.2g}</text>\n'.
                format(0, ytransform(y) + .37 * Fontsize, Teststring, y))

        f.write('</g>\n')

        f.write('<g clip-path="url(#c)">\n'.format(Linewidth))
        
        f.write('<path stroke="black" d="')
        for x in range(max(0, int(ceil(bbox[0]))), min(pathlength - 1, int(floor(bbox[2]))), dx):
            f.write('M{:.2f} {:.2f}V{:.2f}'.format(xtransform(x), ytransform(bbox[1]), ytransform(bbox[1]) - 2 * Tickwidth))
            f.write('M{:.2f} {:.2f}V{:.2f}'.format(xtransform(x), ytransform(bbox[3]), ytransform(bbox[3]) + 2 * Tickwidth))
        f.write('"/>\n')

        f.write('<path stroke="black" d="')

        if log_yaxis:
            for e in range(e0, e1+1):
                for i in range(1,10):
                    y = i*10**e;
                    if y<=bbox[1] or y>=bbox[3]: continue
                    f.write('M{:.2f} {:.2f}H{:.2f}'.format(xtransform(bbox[0]), ytransform(y), xtransform(bbox[0]) + (1+(i==1)) * Tickwidth))
                    f.write('M{:.2f} {:.2f}H{:.2f}'.format(xtransform(bbox[2]), ytransform(y), xtransform(bbox[2]) - (1+(i==1)) * Tickwidth))
        else:
            for y in np.arange(sy, bbox[3], dy):
                f.write('M{:.2f} {:.2f}H{:.2f}'.format(xtransform(bbox[0]), ytransform(y), xtransform(bbox[0]) + 2 * Tickwidth))
                f.write('M{:.2f} {:.2f}H{:.2f}'.format(xtransform(bbox[2]), ytransform(y), xtransform(bbox[2]) - 2 * Tickwidth))
        f.write('"/>\n')

        # Draw the light blue boxes for the intervals in the scale path
        f.write('<g fill="blue" fill-opacity=".1">\n')
        for a, b, c, d in _path_bars(sgd):
            assert a[1] == b[1]
            assert b[0] == c[0]
            assert c[1] == d[1]
            assert d[0] == a[0]
            
            x0 = xtransform(max(a[0], bbox[0]))
            y0 = ytransform(min(b[1], bbox[3]))
            x1 = xtransform(min(b[0], bbox[2]))
            y1 = ytransform(max(c[1], bbox[1]))
            if x0 >= x1 or y0 >= y1: continue
            
            f.write('<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}"/>\n'.
                    format(x0, y0, x1 - x0, y1 - y0))
        f.write('</g>\n')

        # Draw the nodes and the bars for the dendrograms
        if verbose:
            print('Draw nodes and dendrograms.')
        node_y = []
        heights_ext = []
        vertlines = np.empty((pathlength, 2, 2))
        marker_x = np.empty(maxnumvertices + pathlength)
        marker_y = np.empty_like(marker_x)
        marker_i = 0
        dot_x = np.empty(maxnumvertices)
        dot_y = np.empty_like(dot_x)
        dot_i = 0
        zerodot_y = np.empty(pathlength)
        graylines = np.empty((pathlength, 2, 2))
        graylines_i = 0
        for i, dendrogram, diameter in zip(count(), sgd.dendrogram,
                                           sgd.diameter):
            if diameter is None:
                heights_ext.append(None)
                vertlines[i] = (i, 0.)
                ny = np.array([zero_vertex_offset])
            else:
                y = np.hstack((diameter, dendrogram[::-1, 2], 0))
                heights_ext.append(y)
                vy = .5 * (y[1:] + y[:-1])
                if maxvertices is None:
                    y_m = y
                    vy_m = vy
                else:
                    y_m = y[:maxvertices + 1]
                    vy_m = vy[:maxvertices]
                    if y_m[-1] > 0:
                        graylines[graylines_i] = ((i, 0.), (i, y_m[-1]))
                        graylines_i += 1
                vertlines[i] = ((i, y_m[0]), (i, y_m[-1]))
                marker_x[marker_i:marker_i + len(y_m)].fill(i)
                marker_y[marker_i:marker_i + len(y_m)] = y_m
                marker_i += len(y_m)
                dot_x[dot_i:dot_i + len(vy_m)].fill(i)
                dot_y[dot_i:dot_i + len(vy_m)] = vy_m
                dot_i += len(vy_m)
                ny = np.hstack((diameter + zero_vertex_offset, vy))
            node_y.append(ny)
            zerodot_y[i] = ny[0]

        # Broad stems for the part of the dendrograms that is neglected (ie. below "maxvertices")
        f.write('<path stroke="#303030" stroke-width="{}" d="'.format(2 * Tickwidth))
        for a, b in graylines:
            assert a[0] == b[0]
            assert a[1] == 0
            
            if xtransform(a[0]) + Tickwidth <= xtransform(bbox[0]) \
                or xtransform(a[0]) - Tickwidth >= xtransform(bbox[2]):
                continue
            
            y0 = ytransform(max(0, bbox[1])) 
            y1 = ytransform(min(b[1], bbox[3]))
            if y0 <= y1: continue
             
            f.write('M{:.2f} {:.2f}V{:.2f}'.format(xtransform(a[0]), y0, y1))
            
        f.write('"/>\n')

        # Vertical stems for the dendrograms
        f.write('<path stroke="black" d="')
        for a, b in vertlines:
            assert a[0] == b[0]

            if a[0] <= bbox[0] or a[0] >= bbox[2] : continue

            y0 = ytransform(max(b[1], bbox[1])) 
            y1 = ytransform(min(a[1], bbox[3]))
            if y0 <= y1: continue
            
            f.write('M{:.2f} {:.2f}V{:.2f}'.format(xtransform(a[0]), y0, y1))
        f.write('"/>\n')

        # Horizontal tickmarks at interval boundaries
        f.write('<path stroke="black" d="')
        for a, b in zip(marker_x[:marker_i], marker_y[:marker_i]):

            if xtransform(a) + Tickwidth <= xtransform(bbox[0]) or xtransform(a) - Tickwidth >= xtransform(bbox[2]) : continue
            if b <= bbox[1] or b >= bbox[3]: continue
            
            f.write('M{:.2f} {:.2f}H{:.2f}'.format(max(xtransform(a) - Tickwidth, xtransform(bbox[0])),
                                                   ytransform(b),
                                                     min(xtransform(a) + Tickwidth, xtransform(bbox[2]))))
        f.write('"/>\n')

        # Draw dots for the graph nodes 
        f.write('<g fill="red">\n')
        for x, y in zip(dot_x[:dot_i], dot_y[:dot_i]):
            
            if xtransform(x) + Tickwidth <= xtransform(bbox[0]) or xtransform(x) - Tickwidth >= xtransform(bbox[2]) : continue
            if ytransform(y) + Tickwidth <= ytransform(bbox[3]) or ytransform(y) - Tickwidth >= ytransform(bbox[1]) : continue
            
            f.write('<circle cx="{:.2f}" cy="{:.2f}" r="{}"/>\n'.
                    format(xtransform(x), ytransform(y), Tickwidth))
        f.write('</g>\n')

        if infinite_edges and sgd.edges:
            f.write('<g fill="#0050FF">\n')
            for x, y in enumerate(zerodot_y):

                if xtransform(x) + Tickwidth <= xtransform(bbox[0]) or xtransform(x) - Tickwidth >= xtransform(bbox[2]) : continue
                if ytransform(y) + Tickwidth <= ytransform(bbox[3]) or ytransform(y) - Tickwidth >= ytransform(bbox[1]) : continue

                f.write('<circle cx="{:.2f}" cy="{:.2f}" r="{}"/>\n'.
                        format(xtransform(x), ytransform(y), Tickwidth))
            f.write('</g>\n')

        # Draw the edges
        if verbose and sgd.edges:
            print('Fill edge array.')

        NE = sum(map(len, sgd.edges))
        L = np.empty((NE, 2, 2))
        C = np.empty(NE, bool)
        j = 0
        for i, E in enumerate(sgd.edges):
            EE = np.array(E)  # columns: source, target, infinite
            n = np.alen(EE)
            y1 = node_y[i]
            y2 = node_y[i + 1]
            L[j:j + n, :, 0] = (i, i + 1)
            L[j:j + n, 0, 1] = y1[EE[:, 0]]
            L[j:j + n, 1, 1] = y2[EE[:, 1]]
            C[j:j + n] = EE[:, 2]
            j += n

        if len(L):
            L1 = L[:j][C[:j]]
            L2 = L[:j][np.logical_not(C[:j])]

            f.write('<path stroke="black" stroke-width="{}" d="'.
                    format(.2 * Linewidth))
            for (x0, y0), (x1, y1) in L2:
                
                if x1 <= bbox[0] or x0 >= bbox[2] or min(y0, y1) <= bbox[1] or max(y0, y1) >= bbox[3]: continue
                
                f.write('M{:.2f} {:.2f}L{:.2f} {:.2f}'.format(xtransform(x0), ytransform(y0), xtransform(x1), ytransform(y1)))
            f.write('"/>\n')

            if infinite_edges and len(L1):
                f.write('<path stroke="red" stroke-width="{}" d="'.
                        format(.2 * Linewidth))
                for (x0, y0), (x1, y1) in L2:

                    if x1 <= bbox[0] or x0 >= bbox[2] or min(y0, y1) <= bbox[1] or max(y0, y1) >= bbox[3]: continue

                    f.write('M{:.2f} {:.2f}L{:.2f} {:.2f}'.format(xtransform(x0), ytransform(y0), xtransform(x1), ytransform(y1)))
                f.write('"/>\n')

        # Draw the green path with minimal vertical displacement 
        SP = shortest_scale_path(sgd)
        f.write('<path stroke="#00C000" stroke-width="{}" d="'.
                format(2 * Linewidth))
        letter = 'M'
        for x, y in zip(count(0, .5), SP):
            if x + .5 <= bbox[0] or x - .5 >= bbox[2]: continue
            f.write('{}{:.2f} {:.2f}'.format(letter, xtransform(x), ytransform(y)))
            letter = 'L'
        f.write('"/>\n')

        f.write('</g>\n'
                '<use xlink:href="#r"/>\n'
                '</g>\n'
                '</svg>\n')

        # Return the bounding box (plus None for no node size).
        # return (xlim[0], ylim[0], xlim[1], ylim[1], None)

def _path_bars(sgd):
    '''A generator!'''
    for i, num_clust in enumerate(sgd.path):
        if num_clust > 0:
            LB, UB = sgd.layerdata(i)[1:3]
            x1, x2 = i - .6, i + .6
            y1, y2 = UB[num_clust - 1], LB[num_clust]
            if y1 != y2:
                assert y1 > y2
                yield(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

def shortest_scale_path(sgd):
    '''Find the optimal path (i.e. with minimal vertical movement) through the
    intervals corresponding to the scale graph path.

    Anchors are set with step width 1/2 (so that there is an achor for every
    level set and one for each intersection of adjacent level sets).
    '''
    pathlength = len(sgd.path)
    UB = np.empty(pathlength)
    LB = np.empty(pathlength)

    for i, num_clust in enumerate(sgd.path):
        LBs, UBs = sgd.layerdata(i)[1:3]
        if num_clust == 0:
            UB[i] = np.inf
        else:
            UB[i] = UBs[num_clust - 1]
        LB[i] = LBs[num_clust]

    UB[np.isposinf(UB)] = np.max(UB[np.isfinite(UB)])

    UB_ = np.empty(2 * pathlength - 1)
    LB_ = np.empty_like(UB_)

    UB_[::2] = UB
    LB_[::2] = LB

    UB_[1::2] = np.minimum(UB[:-1], UB[1:])
    LB_[1::2] = np.maximum(LB[:-1], LB[1:])

    UB_[1::2], LB_[1::2] = np.maximum(UB_[1::2], LB_[1::2]), \
        np.minimum(UB_[1::2], LB_[1::2])

    return shortest_path(LB_, UB_)
