#! /usr/bin/env python
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

-------------------------------------------------------------------------------

Python PDF generator

The basis for this code was the "text to pdf converter" by Anand B Pillai
<abpillai at gmail dot com>. This program is available at:

    http://code.activestate.com/recipes/532908-text-to-pdf-converter-rewrite/
'''
import sys
import time
import zlib
from collections import OrderedDict

__all__ = ['OnePagePdf', 'PDFPath', 'CourierFont']

if sys.hexversion < 0x03000000:
    ordereddictitems = OrderedDict.iteritems
else:
    ordereddictitems = OrderedDict.items

def asBytes(s):
    if isinstance(s, bytes):
        return s
    elif isinstance(s, str):
        return enc(s)
    elif isinstance(s, int):
        return enc(str(s))
    elif isinstance(s, float):
        if abs(s) < 1:
            return enc('{:.3f}'.format(s))
        else:
            return enc('{:.2f}'.format(s))
    else:
        return s._as_bytes()

def enc(s):
    return s.encode('ascii')

def asciiOrUtf16(s):
    try:
        return enc(s)
    except UnicodeEncodeError:
        # Required by PDF specification: UTF-16 big endian with byte order mark
        return b'\xfe\xff' + s.encode('utf_16_be')

class Array(list):
    def _as_bytes(self):
        return b'[' + b' '.join(map(asBytes, self)) + b']'

class Dictionary(OrderedDict):
    def _as_bytes(self):
        return b"<<\n" \
        + b''.join([b"/" + asBytes(k) + b' ' + asBytes(v) + b"\n"
                    for k, v in ordereddictitems(self)]) \
        + b">>"

class DictionaryObject(Dictionary):
    def __init__(self, objects):
        Dictionary.__init__(self)
        objects.append(self)
        self._number = len(objects)

    def _as_bytes(self):
        return asBytes(self._number) + b" 0 obj\n" \
            + Dictionary._as_bytes(self) \
            + b"\nendobj\n"

    def reference(self):
        return str(self._number) + " 0 R"

class StreamObject(DictionaryObject):
    def __init__(self, objects, compress=True):
        DictionaryObject.__init__(self, objects)
        self._stream = b''
        self._compress = compress
        if compress:
            self['Filter'] = '/FlateDecode'

    def _as_bytes(self):
        if self._compress:
            stream = zlib.compress(self._stream)
        else:
            stream = self._stream
        self['Length'] = len(stream)
        return asBytes(self._number) + b" 0 obj\n" \
            + Dictionary._as_bytes(self) \
            + b"\nstream\n" \
            + stream \
            + b"\nendstream\nendobj\n"

    def append(self, stream):
        self._stream += asBytes(stream)

    def __len__(self):
        return len(self._stream)

class containsXObjectReferences(object):
    def addXObjectReference(self, xo):
        if 'XObject' not in self.Resources:
            self.Resources['XObject'] = ReferenceDictionary()
        if isinstance(xo, XObject):
            xo = [xo]
        for x in xo:
            assert isinstance(x, XObject)
            self.Resources['XObject'].add(x)

class containsExtGStateReferences(object):
    def addExtGStateReference(self, gs):
        if 'ExtGState' not in self.Resources:
            self.Resources['ExtGState'] = ReferenceDictionary()
        if isinstance(gs, ExtGState):
            gs = [gs]
        for x in gs:
            assert isinstance(x, ExtGState)
            self.Resources['ExtGState'].add(x)

class containsFontReferences(object):
    def addFontReference(self, f):
        if 'Font' not in self.Resources:
            self.Resources['Font'] = ReferenceDictionary()
        if isinstance(f, Font):
            f = [f]
        for x in f:
            assert isinstance(x, Font)
            self.Resources['Font'].add(x)
        if 'ProcSet' not in self.Resources:
            self.Resources['ProcSet'] = Array()
        if '/Text' not in self.Resources['ProcSet']:
            self.Resources['ProcSet'].append('/Text')

class XObject(StreamObject, containsXObjectReferences,
              containsExtGStateReferences,
              containsFontReferences):
    def __init__(self, Ob, bbox, XObjects=[], compress=True):
        StreamObject.__init__(self, Ob, compress=compress)
        self['Type'] = '/XObject'
        self['Subtype'] = '/Form'
        # self['FormType'] = '1'
        self['BBox'] = Array(bbox)
        # self['Matrix'] = Array([1,0,0,1,0,0])

        Resources = Dictionary()
        Resources['ProcSet'] = Array(['/PDF'])
        if XObjects:
            Resources['XObject'] = ReferenceDictionary(XObjects)
        self['Resources'] = Resources

    def name(self):
        return 'X' + str(self._number)

class ReferenceDictionary(Dictionary):
    def __init__(self, objects=[]):
        Dictionary.__init__(self)
        for o in objects:
            self.add(o)

    def add(self, o):
        if o.name() not in self:
            self[o.name()] = o.reference()

class ExtGState(DictionaryObject):
    def __init__(self, Objects):
        DictionaryObject.__init__(self, Objects)

    def name(self):
        return 'G' + str(self._number)

class Trailer(Dictionary):
    def __init__(self, xref):
        Dictionary.__init__(self)
        self._xref = xref

    def _as_bytes(self):
        return b"trailer\n" \
            + Dictionary._as_bytes(self) \
            + b"\nstartxref\n" \
            + asBytes(self._xref) \
            + b"\n"

class OnePagePdf(containsXObjectReferences, containsExtGStateReferences,
                 containsFontReferences):
    """ Text2pdf converter in pure Python """

    def __init__(self, ofile, title='', subject='', author='',
                 creator=None, producer=None,
                 keywords=[],
                 height=792, width=612,
                 compress=True):
        #
        self._appname = "Python Mapper"
        # Output file
        self._ofile = ofile
        # Subject
        self._title = asciiOrUtf16(title)
        # Subject
        self._subject = asciiOrUtf16(subject)
        # Author
        self._author = asciiOrUtf16(author)
        # Keywords
        self._keywords = asciiOrUtf16(" ".join(keywords))  # todo

        if creator is None:
            self._creator = asciiOrUtf16(
                self._appname + 
                u" (© Daniel Müllner and Aravindakshan Babu)")
        else:
            self._creator = asciiOrUtf16(creator)
        if producer is None:
            self._producer = self._creator
        else:
            self._producer = asciiOrUtf16(producer)

        # page ht
        self._pageHt = height
        # page wd
        self._pageWd = width
        # Compression?
        self._compress = compress

        # file position marker
        self.Objects = []

        self.Resources = Dictionary()
        self.Content = StreamObject(self.Objects, compress=self._compress)

    def writestr(self, s):
        """ Write string to output file descriptor.
        All output operations go through this function.
        We keep the current file position also here"""
        if not isinstance(s, bytes):
            s = asBytes(s)
        # update current file position
        self._fpos += len(s)
        self._ofs.write(s)

    def addContent(self, content):
        if len(self.Content):
            self.Content.append('\n')
        self.Content.append(content)

    def generate(self, **kwargs):
        """ Perform the actual conversion """
        with open(self._ofile, 'wb') as self._ofs:
            print('Writing pdf file {} ...'.format(self._ofile))
            self.writePDF(**kwargs)
            print('Wrote file {}'.format(self._ofile))

    def writePDF(self):
        """Write the PDF header"""
        Info = DictionaryObject(self.Objects)
        Catalog = DictionaryObject(self.Objects)
        Pages = DictionaryObject(self.Objects)
        Page = DictionaryObject(self.Objects)

        if self._title:
            Info["Title"] = b"(" + self._title + b")"
        if self._subject:
            Info["Subject"] = b"(" + self._subject + b")"
        if self._author:
            Info["Author"] = b"(" + self._author + b")"
        if self._keywords:
            Info["Keywords"] = b"(" + self._keywords + b")"
        Info["Creator"] = b"(" + self._creator + b")"
        Info["Producer"] = b"(" + self._producer + b")"
        t = time.localtime()
        timestr = time.strftime("D:%Y%m%d%H%M%S", t)
        Info["CreationDate"] = "(" + timestr + ")"

        Catalog['Type'] = '/Catalog'
        Catalog['Pages'] = Pages.reference()

        Pages['Type'] = '/Pages'
        Pages['Count'] = '1'
        Pages['MediaBox'] = Array([0, 0, self._pageWd, self._pageHt])
        Pages['Kids'] = Array([Page.reference()])

        if 'ProcSet' not in self.Resources:
            self.Resources['ProcSet'] = Array(['/PDF'])
        else:
            self.Resources['ProcSet'].append('/PDF')

        Page['Type'] = '/Page'
        Page['Parent'] = Pages.reference()
        Page['Contents'] = self.Content.reference()
        Page['Resources'] = self.Resources

        ws = self.writestr
        self._fpos = 0
        ws("%PDF-1.4\n")
        locations = []
        for o in self.Objects:
            locations.append(self._fpos)
            ws(o)
        size = len(locations) + 1

        Tr = Trailer(self._fpos)
        Tr['Size'] = size
        Tr['Root'] = Catalog.reference()
        Tr['Info'] = Info.reference()

        ws("xref\n"
           "0 {}\n"
           "0000000000 65535 f \n".format(size))
        for loc in locations:
            ws("{:010} 00000 n \n".format(loc))
        ws(Tr)
        ws("%%EOF\n")

    def addXObject(self, bbox, stream, XObjects=[], compress=None):
        if compress is None:
            compress = self._compress
        obj = XObject(self.Objects, bbox, XObjects=XObjects, compress=compress)
        obj.append(stream)
        return obj

    def addExtGState(self):
        return ExtGState(self.Objects)

    def addFont(self, fontclass):
        return fontclass(self.Objects)

    miterjoin = 0
    roundjoin = 1
    beveljoin = 2

    buttcap = 0
    roundcap = 1
    projsqcap = 2

    def setlinewidth(self, lw):
        self.addContent('{} w'.format(lw))

    def setlinejoin(self, lj):
        self.addContent('{} j'.format(lj))

    def setlinecap(self, lc):
        self.addContent('{} J'.format(lc))

    def stroke(self, path):
        assert isinstance(path, PDFPath)
        self.addContent(path.stream + 'S')

    def fill(self, path):
        assert isinstance(path, PDFPath)
        self.addContent(path.stream + 'f')

    def clip(self, path):
        assert isinstance(path, PDFPath)
        self.addContent(path.stream + 'W')

    def gsave(self):
        self.addContent('q')

    def grestore(self):
        self.addContent('Q')

    def setfillrgbcolor(self, r, g, b):
        self.addContent('{} {} {} rg'.format(r, g, b))

    def setstrokergbcolor(self, r, g, b):
        self.addContent('{} {} {} RG'.format(r, g, b))

    def setstrokegray(self, x):
        self.addContent('{} G'.format(x))

    def setfillopacity(self, ca):  # todo repetitions
        gs = self.addExtGState()
        gs['ca'] = ca
        self.addExtGStateReference(gs)
        self.addContent('/{} gs'.format(gs.name()))

    def putXObject(self, obj, x, y):
        self.addContent('q 1 0 0 1 {:.2f} {:.2f} cm /{} Do Q'.format(
            x, y, obj.name()))
        self.addXObjectReference(obj)

    def begintext(self):
        self.addContent('BT')

    def endtext(self):
        self.addContent('ET')

    def selectfont(self, font, size):
        self.addContent('/{} {} Tf'.format(font.name(), size))
        self.addFontReference(font)

    def putstring(self, text, x, y):
        self.addContent('1 0 0 1 {:.2f} {:.2f} Tm ({})\''.format(
            x, y, text))

class PDFPath:
    def __init__(self):
        self.stream = ''

    def rectangle(self, x0, y0, x1, y1):
        self.stream += '{:.2f} {:.2f} {:.2f} {:.2f} re '.format(
            min(x0, x1), min(y0, y1), abs(x0 - x1), abs(y0 - y1))

    def moveto(self, x, y):
        self.stream += '{:.2f} {:.2f} m '.format(x, y)

    def lineto(self, x, y):
        self.stream += '{:.2f} {:.2f} l '.format(x, y)

class Font(DictionaryObject):
    def __init__(self, objects):
        DictionaryObject.__init__(self, objects)
        self['Type'] = '/Font'

class CourierFont(Font):
    def __init__(self, objects):
        Font.__init__(self, objects)
        self['Subtype'] = '/Type1'
        self['BaseFont'] = '/Courier'
        self['Name'] = '/{}'.format(self.name())

    def name(self):
        return 'F' + str(self._number)

    def mathaxis(self, fontsize):
        return .28 * fontsize

    def width(self, text, fontsize):
        return .6 * fontsize * len(text)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        sys.exit('Error: input file argument missing')
    elif len(args) > 2:
        sys.exit('Error: Too many arguments')
    ofile = args[1]

    pdfpage = OnePagePdf(ofile, width=300, height=700,
                       title='Test', compress=False)

    stream = (
        '0 29 m 0 13 13 0 29 0 c 45 0 58 13 58 29 c 58 45 45 58 29 58 c 13 58 0 45 0 29 c f')
    obj1 = pdfpage.addXObject([0, 0, 58, 58], stream)

    pdfpage.setlinewidth(10)
    pdfpage.setstrokergbcolor(1, .5, 0)
    pdfpage.setfillrgbcolor(1, 0, 1)
    path = PDFPath()
    path.moveto(0, 0)
    path.lineto(200, 600)
    pdfpage.stroke(path)

    pdfpage.putXObject(obj1, 0, 0)
    pdfpage.putXObject(obj1, 200, 200)

    pdfpage.generate()
