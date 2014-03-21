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

# Small helper class

__all__ = ['progressreporter']

class call_callback:
    def __init__(self, callback):
        self.callback = callback
        self.oldp = -1

    def __call__(self, p):
        if p>self.oldp:
            self.oldp = p
            self.callback(p)

def noop(*args):
    pass

def progressreporter(callback=None):
    if callback:
        return call_callback(callback)
    else:
        return noop
