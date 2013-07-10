#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

if sys.hexversion < 0x03000000: # uniform unicode handling for both Python 2.x and 3.x
    def u(x):
        return x.decode('utf-8')
else:
    def u(x):
        return x
u('''
  cmappertools: Tools for the Python Mapper in C++

  Copyright © 2012 Daniel Müllner
  <http://www.danifold.net>
''')
#import distutils.debug
#distutils.debug.DEBUG = 'yes'
from numpy.distutils.core import setup, Extension

with open('configure.ac', 'r') as f:
    for line in f:
        if line.find('AC_INIT')==0:
            version = line.split('[')[2].split(']')[0]
            break

print('Version: ' + version)

try:
    '''Try to use configure to find out which name to use for the boost::thread
    library.'''
    import subprocess
    import os.path
    if subprocess.call([os.path.realpath('configure')])==0:
        with open('extra_compile_args.txt', 'r') as f:
            extra_compile_args = f.read().strip().split()
            print("Extra compile args: {0}".format(extra_compile_args))
        with open('extra_link_args.txt', 'r') as f:
            extra_link_args = f.read().strip().split()
            print("Extra link args: {0}".format(extra_link_args))
except:
    '''If this does not work, guess.'''
    extra_compile_args = ['-pthread']
    extra_link_args = ['-lboost_thread', '-lboost_chrono']
    with open('config.h', 'w') as f:
        f.write('/* Define to the version of this package. */\n'
                '#define PACKAGE_VERSION "{}\n'.format(version))

setup(name='cmappertools',
      version=version,
      py_modules=[],
      description=('Optional helper module for the Python Mapper package '
                   'with fast, parallel C++ algorithms.'),
      long_description=('Optional helper module for the Python Mapper package '
                        'with fast, parallel C++ algorithms.'),
      ext_modules=[Extension('cmappertools',
                             ['cmappertools.cpp'],
                  # Compiler switches for optimization for the GCC compiler
                  # were generated by the "configure" script.
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args,
                  # (no -pedantic -Wextra)
                             )],
      keywords=['Mapper'],
      author=u("Daniel Müllner"),
      author_email="muellner@math.stanford.edu",
      license="GPLv3 <http://www.gnu.org/licenses/gpl.html>",
      classifiers = [
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
        #"Development Status :: 5 - Production/Stable"
        ],
      url = 'http://math.stanford.edu/~muellner',
      )