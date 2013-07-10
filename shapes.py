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
Test shapes for Mapper
'''
import numpy as np

def circle(samples = 200):
  '''
  Circle
  '''
  phi = np.random.rand(samples,1)*2*np.pi
  return np.hstack((np.cos(phi), np.sin(phi)))

def torus(samples=1000, rminor=.4):
  '''
  Generate n points on a 2-torus

  '''
  M = int(round(np.sqrt(samples)))
  N = int(round(samples/float(M)))
  u = np.linspace(0, 2*np.pi, M, endpoint=False)
  v = np.linspace(0, 2*np.pi, N, endpoint=False).reshape(N,1)
  #v = np.expand_dims(u, 1)

  X = np.empty((M*N,3))
  X[:,0] = ((1. + rminor*np.cos(u)) * np.cos(v)).flat
  X[:,1] = ((1. + rminor*np.cos(u)) * np.sin(v)).flat
  X[:,2] = np.tile(rminor*np.sin(u), N)
  return X
