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
Find an optimal path through a fence of vertical intervals. "Optimal" means the
shortest path from left to right with minimal vertical movement.

Run this as a standalone Python script to see the algorithm in action on random
data. This needs matplotlib for the display.

Otherwise, import the module and use 'shortest_path' on your own data.
'''

from collections import deque
from operator import lt, gt
import numpy as np

__all__ = ['shortest_path']

def shortest_path(LB, UB, verbose=False):
    '''
    Find an optimal path through a fence of vertical intervals. "Optimal" means
    the shortest path from left to right with minimal vertical movement.

    Intervals are equally spaced for simplicity, but that's not necessary
    for the algorithm to work.

    Arguments:

    LB, UB : lower and upper bounds for the path. Must be sequences of numbers
             with equal length.

    Time and space complexity: \Theta(len(LB))
    '''
    pathlength = len(LB)
    assert len(UB) == pathlength, \
        "Upper and lower bounds must have the same length."

    x0 = 0
    y0 = _startpos(LB, UB)

    if isinstance(y0, tuple):
        if verbose:
            print('Shortest path is not unique.\nAny horizontal line '
                  'between {} and {} is optimal.'.format(*y0))
        return np.tile(np.mean(y0), pathlength)

    Y = np.empty(pathlength) # result

    # Convex hulls of the lower and upper bounds. The elements are triples
    # (x, y, slope from the last point to the current point).
    convex_hull_lower = deque([(x0, y0, None)])
    convex_hull_upper = deque([(x0, y0, None)])

    # In each step, introduce one more vertical interval [lb, ub] at the next
    # position x and extend the path as far as its nodes can be unambiguously
    # determined at this stage.
    for x in range(1, pathlength):
        lb = LB[x]
        ub = UB[x]

        assert lb <= ub, \
            "Lower bounds must be less than or equal to the upper bounds."

        # Update the convex hulls with the lower and upper bounds at x.
        _update_hull(convex_hull_lower, x, lb, lt)
        _update_hull(convex_hull_upper, x, ub, gt)

        # When then current upper bound forces the path to touch
        # convex_hull_lower, extend the path.
        #
        # Complexity argument: for every iteration of the while loop, the path
        # Y makes progress, so there are globally at most "pathlength"
        # iterations in the four while loops below.
        while len(convex_hull_lower) > 1 and \
              ub - y0 <= (x - x0) * convex_hull_lower[1][2]:
            x0, y0 = _anchor(x0, y0, convex_hull_lower, convex_hull_upper,
                             x, Y, ub)

        # When then current lower bound forces the path to touch
        # convex_hull_upper, extend the path
        while len(convex_hull_upper) > 1 and \
              lb - y0 >= (x - x0) * convex_hull_upper[1][2]:
            x0, y0 = _anchor(x0, y0, convex_hull_upper, convex_hull_lower,
                             x, Y, lb)

    # Add the last path segments when it touches convex_hull_lower.
    while len(convex_hull_lower) > 1 and convex_hull_lower[1][2] >= 0:
        x0, y0 = _anchor(x0, y0, convex_hull_lower, convex_hull_upper,
                         x, Y, ub)

    # Add the last path segments when it touches convex_hull_upper.
    while len(convex_hull_upper) > 1 and convex_hull_upper[1][2] <= 0:
        x0, y0 = _anchor(x0, y0, convex_hull_upper, convex_hull_lower,
                         x, Y, lb)

    # The final path segments are horizontal.
    Y[x0:] = y0

    return Y

def _update_hull(hull, x, b, cmp):
    '''
    Update the convex hull with the new bound b

    The parameter cmp is either the "less than" or "greater than" operator.

    The amortized time complexity is constant since for every extra iteration
    of the "while" loop, the array "hull" is shortened by one.
    '''
    while True:
        xc, yc, sl = hull[-1]
        if len(hull) == 1 or cmp(b - yc, (x - xc) * sl): break
        hull.pop()
    hull.append((x, b, (b - yc) / float(x - xc)))

def _anchor(x0, y0, hullA, hullB, x, Y, b):
    '''
    Extend the path in Y so that it touches hullA at the second point hullA[1].
    Update hullA and hullB so that both start at (x1, y1),
    '''
    x1, y1, slope = hullA[1]
    Y[x0:x1] = y0 + slope * np.arange(x1-x0)

    hullA.popleft()

    hullB.clear()
    hullB.append((x1, y1, None))
    if x != x1:
        hullB.append((x, b, (b - y1) / float(x - x1)))

    return x1, y1

def _startpos(LB, UB):
    '''
    Find the vertical start position of the optimal path. If the path is not
    unique, then the optimal paths are straight horizontal lines. In this
    case, return a tuple with the minimal and maximal y-coordinate for these
    straight lines.
    '''
    lb = -float('inf')
    ub = float('inf')

    for lbb, ubb in zip(LB, UB):
        if ubb <= lb:
            return lb
        if lbb >= ub:
            return ub
        lb = max(lb, lbb)
        ub = min(ub, ubb)

    if ub == lb:
        return ub
    return (lb, ub)

if __name__ == '__main__':
    '''
    Generate random data points and demonstrate the shortest path algorithm
    visually. The 'matplotlib' package is needed for displaying the function
    graphs.
    '''
    import random
    import math

    seed = random.randint(0, 1e10)
    print('Random seed: {}'.format(seed))
    random.seed(seed)

    # Random data: superposition of K sine waves, sampled at N points.
    N = 50
    K = 5
    fmin = .5
    fmax = 10

    slope = random.uniform(-.3, .3)
    freq = [random.uniform(fmin, fmax) for k in range(K)]
    phase = [random.uniform(0, 2 * math.pi) for k in range(K)]
    amplitude = [random.uniform(0, 1 / freq[k]) for k in range(K)]
    error = [random.uniform(0, .4 / freq[k]) for k in range(K)]
    errorphase = [random.uniform(0, 2 * math.pi) for k in range(K)]

    LB = []
    UB = []

    for x in range(N):
        v = 0
        dv = 0
        for k in range(K):
            v += math.sin(x * freq[k] * 2 * math.pi / N + phase[k]) \
                 * amplitude[k] + slope * x / N
            dv += abs(math.sin(x * freq[k] * 2 * math.pi / N + errorphase[k]) \
                      * error[k])

        LB.append(v - dv)
        UB.append(v + dv)

    Y = shortest_path(LB, UB)

    import matplotlib.pyplot as plt

    plt.plot(range(len(LB)), LB, 'b-+')
    plt.plot(range(len(UB)), UB, 'r-+')
    plt.plot(range(len(Y)), Y, 'k-+')
    plt.show()
