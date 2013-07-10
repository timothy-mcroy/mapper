# -*- coding: utf-8 -*-
r'''
..
  This file is part of the Python Mapper package, an open source tool
  for exploration, analysis and visualization of data.

  Copyright 2011–2013 by the authors:
      Daniel Müllner, http://math.stanford.edu/~muellner
      Aravindakshan Babu, anounceofpractice@hotmail.com

  Python Mapper is distributed under the GPLv3 license. See the project
  home page

      http://math.stanford.edu/~muellner/mapper

  for more information.

  -----------------------------------------------------------------------------

Cover methods
=============

The module ``mapper.cover`` contains classes for various covers of the filter range. Each such class provides an iterator over the levels and a method ``data_index`` which returns the indices of the data points belonging to a certain level.
'''
import itertools
import numpy as np
import collections
import sys
if sys.hexversion < 0x03000000:
    range = xrange
import mapper.tools as tools

__all__ = ['cube_cover_primitive', 'balanced_cover_1d', 'subrange_decomposition_cover_1d']

intervals_default = 10
overlap_default = 50

class level:
    '''
    Data structure for a filter level.

    @param index: The level index
    @type index: tuple
    @param range_min: Minima of the filter coordinates
    @type range_min: numpy.ndarray(dim, dtype=float)
    @param range_max: Maxmima of the filter coordinates
    @type range_max: C{numpy.ndarray(dim, dtype=float)}
    '''
    def __init__(self, index, range_min, range_max):
        self.index = index;
        self.range_min = range_min;
        self.range_max = range_max;

    def __str__( self ):
        return("Filter level. Index: {0}, Range min: {1}, Range max: {2}".\
                   format(self.index, self.range_min, self.range_max))

class _generic_cover:
    '''
    Base class for covers. Do not instantiate this class, but define derived
    classes.
    '''
    def __len__(self):
        return  int(self.intervals.prod()) # numpy.int64 → int

    def _check_input(self, intervals, overlap, info):
        '''
        Generic input check. It is recommended that every class for a cover
        calls this function in its C{__init__} method.
        '''
        if isinstance(intervals, (int, np.integer)):
            intervals = [intervals]
        assert isinstance(intervals, (list, tuple, np.ndarray))
        for i in intervals:
            assert isinstance(i, (int, np.integer))
            assert i>0
        self.intervals = np.array(intervals, dtype=np.int)

        if isinstance(overlap, (int, np.integer, float, np.floating)):
            overlap = [overlap]
        assert isinstance(overlap, (list, tuple, np.ndarray))
        for o in overlap:
            assert isinstance(o, (int, np.integer, float, np.floating))
            assert 0<=o<100
        self.fract_overlap = np.array(overlap, dtype=float) / 100.

        assert isinstance(info, collections.MutableMapping)
        self.info = info
        self.info[ "intervals" ] = self.intervals
        self.info[ "fract_overlap" ] =  self.fract_overlap

    def _check_filter(self, filt, mask):
        '''
        Generic check for the filter array.
        '''
        assert isinstance(filt, np.ndarray)
        if filt.ndim == 1:
            filt = filt[:,np.newaxis]
        assert filt.ndim == 2
        self.filt = filt

        if mask is None:
            self.info['mask'] = None
            filt_mask = filt
            self.mask_bool = True
        else:
            assert mask.ndim==1
            assert mask.dtype==np.bool
            assert len(mask)==len(filt)
            # Store the proportion of unmasked points
            self.info['mask'] = float(len(mask)-mask.sum())/len(mask)
            assert 0<self.info['mask']<=1
            filt_mask = filt[mask]
            self.mask_bool = mask

        self.dim = filt.shape[1]
        self.min = filt_mask.min(axis=0)
        self.max = filt_mask.max(axis=0)
        self.range = (self.max - self.min)

        if len(self.intervals)==1:
            self.intervals = np.repeat(self.intervals, self.dim)
        assert self.intervals.shape==(self.dim,)

        if len(self.fract_overlap)==1:
            self.fract_overlap = np.repeat(self.fract_overlap, self.dim)
        assert self.fract_overlap.shape==(self.dim,)

        self.info["dim"] = self.dim
        self.info["min"] = self.min
        self.info["max"] = self.max
        self.info["range"] = self.range

class cube_cover_primitive (_generic_cover):
    '''Primitive multidimensional cover. The patches are cubes, aligned to the
    coordinate axes and distributed on a rectangular grid.

    Parameters :
      intervals : integer or list of integers
        Number of intervals for each filter component

      overlap : float or list of floats
        Percentage of overlap between the intervals

      info : dict
        `(Ask Aravind)`
    '''
    def __init__(self, intervals=intervals_default, overlap=overlap_default,
                 info={}):
        # We could also just use self.__class__.__name__ for the below
        # we avoid this as per:
        # http://google-styleguide.googlecode.com/svn/trunk/pyguide.html?showone=Power_Features#Power_Features
        info["type"] = "cube_cover_primitive"
        self._check_input(intervals, overlap, info)
        info["str"] = self.__str__()

    def __str__(self):
        return 'Hypercube cover. Intervals: {0}. Overlap: {1}'.\
            format(tuple(self.intervals),tuple(self.fract_overlap*100))

    def __iter__(self):
        return self

    def __call__(self, filt, mask=None):
        '''
        Provide the iterator over the levels. In the present case, the
        (multi-)levels are the cartesian product of the levels in each
        coordinate.

        Parameters :
          filt : ndarray((N,n), dtype=float)
            Vector-valued filter function with :math:`n` components on :math:`N` data points.

           mask : boolean array to mask data points
        '''
        self._check_filter(filt, mask)
        self.interval_length = self.range / \
            ( self.intervals - (self.intervals-1)*self.fract_overlap )
        self.step_size = self.interval_length*(1-self.fract_overlap)

        self.iter = itertools.product(*(range(i) for i in self.intervals))
        return self

    def _minmax(self, index):
        range_min = self.min + index * self.step_size
        range_max = range_min + self.interval_length
        return range_min, range_max

    def next(self):
        '''
        Provide the iterator over the levels for Python 2.
        '''
        index = self.iter.next()
        range_min, range_max = self._minmax(index)
        return level(index, range_min, range_max)

    def __next__(self):
        '''
        Provide the iterator over the levels for Python 3.
        '''
        index = self.iter.__next__()
        range_min, range_max = self._minmax(index)
        return level(index, range_min, range_max)

    def data_index(self, level):
        '''
        Return the indices to the data points for a given level identifier.

        Parameters :
          level : ``level``
            the level identifier

        Returns:
          out : ndarray, dtype=int
            data indices
        '''
        levelindex = np.array(level.index)
        range_min, range_max = self._minmax(levelindex)
        # The first and last interval in each coordinate extend to ±infinity,
        # to account for floating-point inaccuracies.
        range_min[np.nonzero(levelindex==0)] = -np.inf
        range_max[np.nonzero(levelindex+1==self.intervals)] = np.inf

        lb = ( self.filt >= range_min )
        ub = ( self.filt <= range_max )
        b = np.logical_and(lb,ub)
        idx, = np.where(np.logical_and(self.mask_bool, b.all(axis=1)))
        return idx

    def cannot_intersect(self, levels):
        '''(Experimental)'''
        return np.any(self.step_size*np.ptp(levels, axis=0) > \
                          self.interval_length)

class balanced_cover_1d(_generic_cover):
    '''
    One-dimensional balanced cover. The interval boundaries are distributed so
    that each patch covers the same fraction of the data set.

    Parameters :
      intervals : int
        Number of intervals for each filter component

      overlap : float
        Percentage of overlap between the intervals

      info : dict
        `(Ask Aravind)`
    '''
    def __init__(self, intervals=intervals_default, overlap=overlap_default,
                 info={}):
        info["type"] = "balanced_cover_1d"
        self._check_input(intervals, overlap, info)
        info["str"] = self.__str__()

    def __str__(self):
        return 'Balanced cover with {0} intervals and {1}% overlap.'.\
            format(int(self.intervals[0]), self.fract_overlap[0]*100)

    def __iter__(self):
        return self

    def __call__(self, filt, mask=None):
        '''
        Provide the iterator over the levels.
        '''
        self._check_filter(filt, mask)
        # This cover method is only for one-dimensional filter functions.
        assert(self.dim==1)
        # The interval length measures indices, not filter values
        # in this case.
        self.interval_length = 1. / \
            ( self.intervals[0] - (self.intervals[0]-1)*self.fract_overlap )
        self.step_size = self.interval_length*(1-self.fract_overlap)

        if mask is None:
            self.n = len(self.filt)
            self.sortorder = np.argsort(np.ravel(self.filt))
        else:
            idx = np.flatnonzero(mask)
            self.n = len(idx)
            sortorder = np.argsort(np.ravel(self.filt[mask]))
            self.sortorder = idx[sortorder]

        assert len(self.sortorder)==self.n

        self.iter = range(self.intervals[0]).__iter__()
        return self

    def _rangeminmax(self, index):
        '''
        Minimum and maximum filter values for a given level, linearly
        interpolated between two adjacent filter values in the data set
        to give good quantile information.
        '''
        p1 = index*self.step_size
        p2 = p1+self.interval_length
        return self._sample_quantile(p1), self._sample_quantile(p2)

    def _sample_quantile(self, p):
        '''
        Method for the sample quantile. The formula is the same as
        R's 'quantile' method with parameter type=8.

        See
        http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html

        (Type 8 is cited as recommended.)
        '''
        m = (p+1)/3.
        h, = self.n*p+m-1# -1: index-0-based, different from the R
        # documentation, which is index-1-based
        if h<0:
            return self.filt[self.sortorder[0]]
        elif h>=self.n-1:
            return self.filt[self.sortorder[-1]]
        else:
            f, j = np.modf(h)
            return self.filt[self.sortorder[j]] + f * \
                (self.filt[self.sortorder[j+1]]-self.filt[self.sortorder[j]])

    def _indexminmax(self, index):
        '''
        Minimum and maximum sorted filter index for a given level.
        '''
        p1 = index*self.step_size
        p2 = p1+self.interval_length
        return self._quantile_index(p1), self._quantile_index(p2)

    def _quantile_index(self, p):
        m = (p+1.)/3.
        h, = self.n*p+m
        j = np.floor(h)
        return max(0,min(self.n,j))

    def next(self):
        '''
        Provide the iterator over the levels for Python 2.
        '''
        index = (self.iter.next(),)
        return level(index, *self._rangeminmax(index))

    def __next__(self):
        '''
        Provide the iterator over the levels for Python 3.
        '''
        index = (self.iter.__next__(),)
        return level(index, *self._rangeminmax(index))

    def data_index(self, level):
        '''
        Return the indices to the data points for a given level identifier.

        @param level: the level identifier
        @type level: L{level}
        @return: data indices
        @rtype: C{numpy.ndarray( dtype=int )}
        '''
        idx_min, idx_max = self._indexminmax(level.index)
        return self.sortorder[idx_min:idx_max]

    def cannot_intersect(self, levels):
        return np.any(self.step_size*np.ptp(levels, axis=0) > \
                          self.interval_length)

class subrange_decomposition_cover_1d( _generic_cover ):
    # Note: The data_index method will fail if it's called before the covers are iterated thru

    # To be done TBD: respect masks

    def __init__(self, intervals_by_subrange, overlaps_by_subrange, subrange_boundaries, info={}):

        self.info = info
        assert isinstance(info, collections.MutableMapping)
        info["type"] = "subrange_decomposition_cover_1d"

        self.intervals_by_subrange = intervals_by_subrange
        self.overlaps_by_subrange = overlaps_by_subrange
        self.subrange_boundaries = subrange_boundaries

        self.subrange_covers = []
        for intervals, overlaps in itertools.izip( intervals_by_subrange, overlaps_by_subrange ):
            self.subrange_covers.append( cube_cover_primitive( intervals, overlaps ) )
        info["str"] = self.__str__()

    def __len__(self):
        return  sum(self.intervals_by_subrange)

    def __str__(self):
        return 'Subrange decomposition cover 1d. Intervals by subrange: ' + \
            str(tuple(self.intervals_by_subrange)) + \
            '. Overlap percent by subrange: ' + \
            str(tuple(self.overlaps_by_subrange)) +\
            '. Subrange boundaries: '+ str(tuple(self.subrange_boundaries)) + \
            '.'

    def __call__(self, filt ):
        '''
        Provide the iterator over the levels.

        @param filt: filter function
        @type filt: C{numpy.ndarray((n,), dtype=float)}
        @param subrange_boundaries: iterable of floats that specify the subranges. k entries for k-1 subranges.
                          First value is always min(filt), last value is max(filt).
        @type subrange_boundaries: C{numpy.ndarray((k,), dtype=float)}
        '''
        assert isinstance(filt, np.ndarray)
        if filt.ndim == 1:
            filt = filt[:,np.newaxis]
        assert filt.ndim == 2
        self.filt = filt

        self.dim = filt.shape[1]
        self.min = filt.min(axis=0)
        self.max = filt.max(axis=0)
        self.range = (self.max - self.min)

        self.info["dim"] = self.dim
        self.info["min"] = self.min
        self.info["max"] = self.max
        self.info["range"] = self.range

        # This cover method is only for one-dimensional filter functions.
        assert(self.dim==1)

        prev_subcover_step_size = 0.
        prev_subcover_interval_length = 0.
        for subrange, cover in itertools.izip( tools.pairwise(self.subrange_boundaries), self.subrange_covers ):
            subrange_min_pctg, subrange_max_pctg = subrange
            subrange_min = subrange_min_pctg/100. * self.range + self.min
            subrange_max = subrange_max_pctg/100. * self.range + self.min
            adjusted_subrange_min = subrange_min-prev_subcover_interval_length+prev_subcover_step_size
            subrange_filt = filt[np.logical_and(filt>=adjusted_subrange_min, filt<=subrange_max)]
            cover( subrange_filt )
            prev_subcover_interval_length, prev_subcover_step_size = cover.interval_length, cover.step_size

        #self.iter = xrange(sum(intervals_by_subrange))
        return self

    def __iter__(self):
        '''
        Provide the iterator over the levels.
        '''

        for index, l in enumerate( itertools.chain( *self.subrange_covers ) ):
            yield level((index,), l.range_min, l.range_max)

    def data_index(self, level):
        '''
        Return the indices to the data points for a given level identifier.

        @param level: the level identifier
        @type level: L{level}
        @return: data indices
        @rtype: C{numpy.ndarray( dtype=int )}
        '''
        range_min, range_max = level.range_min, level.range_max
        lb = ( self.filt >= range_min )
        ub = ( self.filt <= range_max )
        b = np.logical_and(lb,ub)
        idx, = np.where(b.all(axis=1))
        return idx
