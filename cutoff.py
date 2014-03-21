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
Find a cutoff value for a clustering tree.

Several algorithms are combined into this module.
'''
import numpy as np

__all__ = ['histogram', 'first_gap', 'biggest_gap']

def check_input(heights, diam):
    '''Input checking'''
    assert isinstance(heights, np.ndarray)
    assert heights.ndim == 1
    assert heights.dtype == np.float
    assert np.min(heights) >= 0
    assert isinstance(diam, (float, np.floating))
    assert diam >= np.max(heights)

class histogram:
    """
    Generate a histogram of the heights and look for the first gap.

    Algorithm: The heights in the clustering tree are put into I{num_bins}
    bins in a histogram. The range of the histogram is M{[0,diam]}. The
    algorithm then selects the midpoint of the first empty histogram interval
    as the cutoff value.

    Example:

    heights=[1,1,1,2,3,4], diam = 11.0 and num_bins = 10: cutoff = 5.5, ie we
    have one single cluster.

    heights=[1,1,1,2,3,5], diam = 11.0 and num_bins = 10: cutoff = 4.5, ie we
    have two clusters.

    @param num_bins: number of bins we want to bin heights into
    @type num_bins: integer S{>=}2
    """

    def __init__(self, num_bins):
        assert isinstance(num_bins, (int, np.integer))
        assert num_bins >= 2
        self.num_bins = num_bins

    def __str__(self):
        return 'Histogram method with {0} bins'.format(self.num_bins)

    def __call__(self, heights, diam):
        '''
        @param heights: vector of heights at which the dendogram collapsed
        clusters
        @type heights: nump.ndarray(n, dtype=float)
        @param diam: The diameter of the data set, ie the maximal pairwise
        distance between points.
        @type diam: float S{>=}max(heights)

        @return: number of clusters
        @rtype: int
        '''
        check_input(heights, diam)

        [bin_counts, bin_leftedges] = np.histogram(heights,
                                                   bins=self.num_bins,
                                                   range=(0,diam))
        # indices of the empty bins; don't consider the last bin
        empty_bins_idx, = np.nonzero(bin_counts[:-1] == 0)

        # Find the first gap
        if empty_bins_idx.size:
            min_bin_idx = empty_bins_idx[0]
            num_clust = bin_counts[min_bin_idx:].sum()+1
        else:
            num_clust = 1
        return num_clust

class first_gap:
    '''
    Look for the first gap of size C{gap} or bigger in the
    heights of the clustering tree.

    This is similar to L{cluster_cutoff_histo}
    with M{gap = diam / num_bins}.

    @param gap: gap size
    @type gap: float S{>=}0
    '''
    def __init__(self, gap):
        assert isinstance(gap, (float, np.floating))
        assert 0 < gap < 1
        self.gap = gap

    def __str__(self):
        return 'First gap of relative width {0}'.format(self.gap)

    def __call__(self, heights, diam):
        '''
        @param heights: vector of heights at which the dendogram collapsed
        clusters
        @type heights: nump.ndarray(n, dtype=float)
        @param diam: The diameter of the data set, ie the maximal pairwise
        distance between points.
        @type diam: float S{>=}max(heights)

        @return: number of clusters
        @rtype: int
        '''
        check_input(heights, diam)

        # differences between subsequent elements (and heights[0] at the
        # beginning)
        diff = np.ediff1d(heights, to_begin=heights[0])
        gap_idx, = np.nonzero(diff >= self.gap*diam)
        if gap_idx.size:
            num_clust = heights.size+1-gap_idx[0]
        else:
            # no big enough gap -> one single cluster
            num_clust = 1
        return num_clust

class biggest_gap:
    '''
    Look for the biggest logarithmic gap in the clustering tree, i.e. the
    index where the ratio of consecutive heights in the dendrogram is biggest.
    '''
    def __str__(self):
        return 'Biggest logarithmic gap'

    def __call__(self, heights, diam):
        '''
        @param heights: vector of heights at which the dendogram collapsed
        clusters
        @type heights: nump.ndarray(n, dtype=float)
        @param diam: The diameter of the data set, ie the maximal pairwise
        distance between points.
        @type diam: float S{>=}max(heights)

        @return: number of clusters
        @rtype: int
        '''
        check_input(heights, diam)

        # ratios between subsequent elements
        h2 = np.hstack((heights[1:], diam))
        ratios = h2 / heights
        idx = ratios.argmax()
        if np.nonzero(ratios==ratios[idx])[0].size>1:
            raise Warning("Ambiguous cutoff value")
        num_clust = ratios.size-idx
        return num_clust

class variable_exp_gap:
    '''
    Look for the biggest gap. Variable exponent.

    @param exp: exponent
    @type exp: float
    '''
    def __init__(self, exponent, maxcluster=None):
        assert isinstance(exponent, (float, np.floating, int, np.integer))
        self.exponent = exponent
        self.maxcluster = maxcluster

    def __str__(self):
        return 'Biggest gap, exponent {0}, max. clusters {1}'.\
            format(self.exponent, self.maxcluster)

    def __call__(self, heights, diam):
        '''
        @param heights: vector of heights at which the dendogram collapsed
        clusters
        @type heights: nump.ndarray(n, dtype=float)
        @param diam: The diameter of the data set, ie the maximal pairwise
        distance between points.
        @type diam: float S{>=}max(heights)

        @return: number of clusters
        @rtype: int
        '''
        check_input(heights, diam)
        OneMinusExp = 1.-self.exponent

        h2 = np.hstack((heights[1:], diam))
        if self.maxcluster is not None:
            heights = heights[-self.maxcluster:]
            h2 = h2[-self.maxcluster:]
        startidx = 0;
        if OneMinusExp>0.:
            diffs = np.power(h2, OneMinusExp) - \
                np.power(heights, OneMinusExp)
        else:
            while heights[startidx]==0.:
                startidx += 1

            if OneMinusExp==0.:
                diffs = np.log(h2[startidx:]) - np.log(heights[startidx:])
            else:
                diffs = np.power(heights[startidx:], OneMinusExp) - \
                    np.power(h2[startidx:], OneMinusExp)

        idx = diffs.argmax()
        if np.nonzero(diffs==diffs[idx])[0].size>1:
            raise Warning("Ambiguous cutoff value")
        num_clust = diffs.size-idx
        return num_clust

class variable_exp_gap2:
    '''
    Look for the biggest gap. Variable exponent, 2nd variant.

    @param exp: exponent
    @type exp: float
    '''
    def __init__(self, exponent, maxcluster=None):
        assert isinstance(exponent, (float, np.floating, int, np.integer))
        self.exponent = exponent
        self.maxcluster = maxcluster

    def __str__(self):
        return 'Biggest gap variant 2, exponent {0}, max. clusters {1}'.\
            format(self.exponent, self.maxcluster)

    def __call__(self, heights, diam):
        '''
        @param heights: vector of heights at which the dendogram collapsed
        clusters
        @type heights: nump.ndarray(n, dtype=float)
        @param diam: The diameter of the data set, ie the maximal pairwise
        distance between points.
        @type diam: float S{>=}max(heights)

        @return: number of clusters
        @rtype: int
        '''
        check_input(heights, diam)

        startidx = 0;
        while heights[startidx]==0.:
            startidx += 1
        heights = heights[startidx:]

        h2 = np.hstack((heights[1:], diam))
        if self.maxcluster is not None:
            heights = heights[-self.maxcluster:]
            h2 = h2[-self.maxcluster:]
        diffs = (h2-heights) * np.power(heights, -self.exponent)

        idx = diffs.argmax()
        if np.nonzero(diffs==diffs[idx])[0].size>1:
            raise Warning("Ambiguous cutoff value")
        num_clust = diffs.size-idx
        return num_clust
