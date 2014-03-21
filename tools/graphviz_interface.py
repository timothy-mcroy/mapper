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
import subprocess
import sys
from mapper.tools import dict_items

__all__ = ['graphviz_node_pos']

def graphviz_node_pos(S, nodes):
    D = dot_from_mapper_output(S, nodes)

    P = dotparser(D)
    P.parse_graph()
    return zip(*[(int(n), tuple(map(float, a['pos'].split(',')))) \
                 for n, a in dict_items(P.nodes)])

    # nodes = re.findall(r'\{\s*node\s\[.*\];\s*([0-9]+)\s*\[(?:.|\s)*?pos="(.*),(.*)"(?:.|\s)*?\];\s*\}', out, re.MULTILINE)
    # nodepos = re.findall(r'\{\s*(?:node\s\[.*?\];\s*)?([0-9]+)\s*\[.*?pos="(.*?),(.*?)".*?\];\s*\}', D, re.DOTALL)
    # nodepos = [(int(n), float(x), float(y)) for n, x, y in nodepos]
    # N  = [n for n, x, y in nodepos]
    # P = [(x, y) for n, x, y in nodepos]

def dot_from_mapper_output(S, nodes):
    '''
    Generate a dot file from Mapper output and process it with Graphviz.
    '''
    if S.dimension < 0:
        return None

    graphvizcommand = 'neato'
    try:
        exception_to_catch = FileNotFoundError
    except NameError:
        exception_to_catch = OSError
    try:
        p = subprocess.Popen([graphvizcommand], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except exception_to_catch:
        sys.stderr.write('Error: Could not call "{0}". '
                         'Make sure that graphviz is installed and that {0} is in the search path.\n'.
                         format(graphvizcommand))
        raise

    p.stdin.write('graph mapper_output { '
                  'node [ shape=circle, label="" ];'.encode('ascii')
                  )
    # Caution: Not all nodes may be vertices!
    vertices = [n for n, in S[0]]
    vertices.sort()

    #f = [float(nodes[i].attribute) for i in vertices]
    #fmin, fmax = min(f), max(f)

    for i, n in enumerate(vertices):
        p.stdin.write('{};'.format(n).encode('ascii'))

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

class dotparser:
    '''
    See the dot language specification at
    
    http://www.graphviz.org/doc/info/lang.html
    '''
    def __init__(self, D):
        self.D = D
        self.pos = 0
        self.len = len(D)
        self.tokenstate = []
        self.nodes = {}

    CURLYOPEN = '{'
    CURLYCLOSE = '}'
    SQUAREOPEN = '['
    SQUARECLOSE = ']'
    SEMICOLON = ';'
    COLON = ':'
    COMMA = ','
    EQUALSIGN = '='
    EOF = None
    ALPH = 0
    NUMBER = 1
    EDGERHS = 2
    QUOTEDID = 3
    HTMLID = 4

    def parse_graph(self):
        # print(self.D)
        
        self.getnexttoken()

        if self.tc == dotparser.ALPH and self.t.lower() == 'strict':
            self.getnexttoken()

        if self.tc == dotparser.ALPH and self.t.lower() == 'digraph':
            self.digraph = True
        elif self.tc == dotparser.ALPH and self.t.lower() == 'graph':
            self.digraph = False
        else:
            raise ValueError('expected graph/digraph statement but got ' + self.t)

        self.getnexttoken()

        self.parse_id()

        if self.tc != dotparser.CURLYOPEN:
            raise ValueError('expected a curly brace but got ' + self.t)
        self.getnexttoken()

        self.parse_stmt_list()

        if self.tc != dotparser.CURLYCLOSE:
            raise ValueError('expected a closing curly bracket but got ' + self.t)

        self.getnexttoken()
        if self.tc != dotparser.EOF:
            raise ValueError('expected end of dot file but got ' + self.t)

        assert not len(self.tokenstate)

    def parse_stmt_list(self):
        is_stmt = True
        while is_stmt:
            is_stmt = self.parse_stmt()
            if self.tc == dotparser.SEMICOLON:
                self.getnexttoken()

    def parse_stmt(self):
        return (self.parse_subgraph() or \
                self.parse_attr_stmt() or \
                self.parse_edge_stmt() or \
                self.parse_node_stmt() or \
                self.parse_id_assignment()[0])
        
    def parse_attr_stmt(self):
        if self.tc == dotparser.ALPH and self.t.lower() in ('graph', 'node', 'edge'):
            self.tsave()
            self.getnexttoken()
            if not self.parse_attr_list()[0]:
                self.trestore()
                return False

            self.tdiscard()
            return True

        return False

    def parse_attr_list(self):
        attrlist = {}
        is_attr_list, attr = self.parse_attr_list_()
        if not is_attr_list:
            return False, None
        
        while is_attr_list:
            for k in attr:
                if k in attrlist:
                    raise KeyError('duplicate attribute name')
            attrlist.update(attr)
            is_attr_list, attr = self.parse_attr_list_()

        return True, attrlist

    def parse_attr_list_(self):
        if self.tc != dotparser.SQUAREOPEN:
            return False, None

        self.tsave()
        self.getnexttoken()
        
        is_a_list, a_list = self.parse_a_list()

        if not is_a_list or self.t != dotparser.SQUARECLOSE:
            self.trestore()
            return False, None

        self.getnexttoken()
        self.tdiscard()
        return True, a_list

    def parse_a_list(self):
        a_list = {}
        is_a_list, id_assignment = self.parse_id_assignment()
        if not is_a_list:
            return False, None

        while is_a_list:
            if id_assignment[0] in a_list:
                raise KeyError('duplicate attribute name')
            a_list[id_assignment[0]] = id_assignment[1]
            if self.tc == dotparser.COMMA:
                self.getnexttoken()

            is_a_list, id_assignment = self.parse_id_assignment()

        return True, a_list

    def parse_id_assignment(self):
        self.tsave()
        
        is_id, id1 = self.parse_id()

        if not is_id or self.tc != dotparser.EQUALSIGN:
            self.trestore()
            return False, None

        self.getnexttoken()

        is_id, id2 = self.parse_id()

        if not is_id:
            self.trestore()
            return False, None

        self.tdiscard()
        return True, (id1, id2)

    def parse_edge_stmt(self):
        self.tsave()

        if not((self.parse_subgraph() or self.parse_node_id()[0]) and self.parse_edge_RHS()):
            self.trestore()
            return False

        self.parse_attr_list()
        self.tdiscard()
        return True

    def parse_edge_RHS(self):
        if not self.parse_edge_RHS_():
            return False

        while self.parse_edge_RHS_():
            pass

        return True

    def parse_edge_RHS_(self):
        if (self.digraph and self.t != '->') or (not self.digraph and self.t != '--'):
            return False

        self.tsave()
        self.getnexttoken()

        if not(self.parse_node_id()[0] or self.parse_subgraph()):
                self.trestore()
                return False

        self.tdiscard()
        return True

    def parse_node_stmt(self):
        is_node_id, node_id = self.parse_node_id()
        if not is_node_id:
            return False
        
        if node_id in self.nodes:
            raise KeyError('duplicate node id')
        
        
        dummy, attr_list = self.parse_attr_list()
        self.nodes[node_id] = attr_list
        return True

    def parse_node_id(self):
        is_id, node_id = self.parse_id()
        if not is_id:
            return False, None

        self.parse_port()
        return True, node_id

    def parse_port(self):
        if self.tc != dotparser.COLON:
            return False

        self.tsave()
        self.getnexttoken()

        if self.parse_id()[0]:
            if self.tc != dotparser.COLON:
                self.getnexttoken()
                if not self.parse_compass_pt():
                    self.trestore()
                    return False
        else:
            if not self.parse_compass_pt():
                self.trestore()
                return False

        self.tdiscard()
        return True

    def parse_subgraph(self):
        self.tsave()

        if self.tc == dotparser.ALPH and self.t.lower() == 'subgraph':
            self.getnexttoken()
            self.parse_id()

        if self.tc != dotparser.CURLYOPEN:
            self.trestore()
            return False

        self.getnexttoken()
        self.parse_stmt_list()

        if self.tc != dotparser.CURLYCLOSE:
            self.trestore()
            return False

        self.getnexttoken()
        self.tdiscard()
        return True

    def parse_compass_pt(self):
        if self.tc == dotparser.ALPH and self.t in ('n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', 'c', '_'):
            self.getnexttoken()
            return True
        else:
            return False

    def parse_id(self):
        if self.tc in (dotparser.ALPH, dotparser.NUMBER, dotparser.QUOTEDID, dotparser.HTMLID):
            id = self.t
            self.getnexttoken()
            return True, id

        return False, None

    def skipwhitespace(self):
        while self.pos < self.len and \
            (self.D[self.pos] in ' \t\n\r\f\v' or \
             (self.D[self.pos] == '\\' and self.pos < self.len - 1 and self.D[self.pos] == '\n')):
            self.pos += 1
            
        if self.pos < self.len and self.D[self.pos] == '/':
            raise RuntimeError('not implemented yet')

        if self.pos < self.len and self.D[self.pos] == '#' and self.pos > 0 and self.D[self.pos - 1] in '\n\r':
            raise RuntimeError('not implemented yet')

    def getnexttoken(self):
        self.skipwhitespace()

        if self.pos == self.len:
            # print ("Token: EOF")
            self.tc = dotparser.EOF
            self.t = None
            return
        
        c = self.getchar()

        if c in '{}[];:,=':
            self.t = self.tc = c
        elif (c >= '0' and c <= '9') or c == '-' or c == '.':
            self.get_edge_rhs_or_number(c)
        elif (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c in '_':
            self.getsimpleid(c)
        elif c == '"':
            self.getquotedid(c)
        elif c == '<':
            self.gethtmlid(c)
        elif c == '+':
            raise RuntimeError('not implemented yet')
        else:
            raise ValueError('unknown token: ' + c)
        # print ("Token: " + self.t)

    def getchar(self):
        if self.pos == self.len:
            raise ValueError('incomplete dot file')
        c = self.D[self.pos]
        self.pos += 1
        return c
        

    def getsimpleid(self, c):
        self.tc = dotparser.ALPH
        self.t = ''
        while (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c in '_':
            self.t += c
            if self.pos == self.len:
                return
            c = self.getchar()
            
        self.pos -= 1


    def getquotedid(self, c):
        self.tc = dotparser.QUOTEDID
        self.t = ''
        while True:
            c = self.getchar()

            if c == '"':
                if len(self.t) and self.t[-1] == '\\':
                    self.t[-1] = c
                else:
                    return
            else:
                self.t += c

    def get_edge_rhs_or_number(self, c):
        self.t = ''

        if c == '-':
            self.t = c
            c = self.getchar()
                            
            if c in ('-', '>'):
                self.t += c
                self.tc = dotparser.EDGERHS
                return
            
        self.tc = dotparser.NUMBER 

        digits = False
        while c >= '0' and c <= '9':
            self.t += c
            digits = True
            if self.pos == self.len:
                return
            c = self.getchar()

        if c == '.':
            self.t += c
            if self.pos == self.len and Digits:
                return
            c = self.getchar()

            while c >= '0' and c <= '9':
                self.t += c
                if self.pos == self.len:
                    return
                c = self.getchar()
                
        self.pos -= 1
        
    def gethtmlid(self, c):
        raise RuntimeError('HTML IDs are not implemented yet')

    def tsave(self):
        self.tokenstate.append((self.pos, self.t, self.tc))

    def trestore(self):
        self.pos, self.t, self.tc = self.tokenstate.pop()

    def tdiscard(self):
        self.tokenstate.pop()
