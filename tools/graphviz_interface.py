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
Interface to graphviz
'''
from mapper.mapper_output import dict_keys, dict_values, dict_items
import subprocess
import sys
import re

__all__ = ['graphviz_node_pos']

def graphviz_node_pos(nodes, S):
    D = dot_from_mapper_output(nodes, S)

    #nodes = re.findall(r'\{\s*node\s\[.*\];\s*([0-9]+)\s*\[(?:.|\s)*?pos="(.*),(.*)"(?:.|\s)*?\];\s*\}', out, re.MULTILINE)
    nodepos = re.findall(r'\{\s*(?:node\s\[.*?\];\s*)?([0-9]+)\s*\[.*?pos="(.*?),(.*?)".*?\];\s*\}', D, re.S)
    nodepos = [(int(n), float(x), float(y)) for n, x, y in nodepos]
    
    return [n for n, x, y in nodepos], [(x, y) for n, x, y in nodepos]

def dot_from_mapper_output(nodes, S):
       
    if S.dimension < 0:
        return None
        
    # todo avoid knots with edge weight
        
    graphvizcommand = 'neato'
    try:
        
        p = subprocess.Popen([graphvizcommand], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        sys.stderr.write('Error: Could not call "{0}". '
                         'Make sure that graphviz is installed and that {0} is in the search path.\n'.
                         format(graphvizcommand))
        raise 
            

    p.stdin.write('graph mapper_output { '
                  #'edge [length=.01];'
                  'node [ shape=circle, label="" ];'.encode('ascii')
                  )      
    # Caution: Not all nodes may be vertices!
    vertices = [n for n, in dict_keys(S[0])]
    vertices.sort()
    
    f = [float(nodes[i].attribute) for i in vertices]
    fmin, fmax = min(f), max(f)
     
    for i, n in enumerate(vertices):
        #p.stdin.write('{{ node [ pos="{0},{1}" ] {2} }}'.format(1000.0*(f[i]-fmin)/(fmax-fmin), 500.0*i/float(len(vertices)), n).
        #              encode('ascii'))
        p.stdin.write('{{ node [  ] {2} }}'.format(1000.0*(f[i]-fmin)/(fmax-fmin), 500.0*i/float(len(vertices)), n).
                      encode('ascii'))

    if S.dimension > 0:
        for (a, b), w in dict_items(S[1]):
            p.stdin.write('{0}--{1};'.format(a, b).encode('ascii'))

    p.stdin.write('}'.encode('ascii'))

    out, err = p.communicate()
    p.stdin.close()
    if err:
        print(err)
        raise RuntimeError(err)
    if p.returncode != 0:
        raise RuntimeError('Graphviz exited with return code ' + p.returncode)

    return out.decode('ascii')