Installation
============

Python Mapper should be easy to install under Linux since all software packages it depends on are widely used and probably included in all major Linux distributions. In principle, Python Mapper does not contain any platform-specific code and depends only on cross-platform packages. Hence, there is no obvious reason why it should not be possible to run it on Mac OS and Microsoft Windows.

So far, installation has been tested under

* Arch Linux
* OS X Mountain Lion

If you install Python Mapper on a certain platform not in the list above, please let me (`Daniel <http://danifold.net>`_) know so that I can extend the list. Especially if you needed to tweak or modify something, I am interested to know about it.

(If you succeed in all installation steps on Windows except compilation of cmappertools, also let me know. I might be able to provide pre-compiled packages for cmappertools. After all, compilation of Python modules on Windows is still a pain.)

Requirements
------------

* `Python <http://www.python.org/>`_ 2.6 or higher. The GUI needs Python 2 since it depends on wxPython and PyOpenGL; Python Mapper itself can be run under Python 2 and Python 3.
* `NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org/>`_
* `Matplotlib  <http://matplotlib.sourceforge.net/>`_
* `Networkx <http://networkx.lanl.gov/>`_
* Optionally cmappertools. Python Mapper will run without this module, but with limited functionality.

  -  cmappertools need the `Boost C++ libraries <http://www.boost.org/>`_.




For the GUI:

* `wxPython <http://www.wxpython.org/>`_
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_

Highly recommended:

* cmappertools

  This is Daniel Müllner's Python module, written in C++, which replaces some slower Python routines by fast, parallelized C++ algorithms. Currently, it is in the ``cmappertools`` subdirectory of the Python Mapper distribution. Compile and install it in the standard way, eg.::

    cd cmappertools
    python setup.py install --user

* `fastcluster <http://math.stanford.edu/~muellner/fastcluster.html>`_

Download
--------

The source distribution of Python Mapper can be downloaded here:

.. admonition:: Download link for Python Mapper

   http://math.stanford.edu/~muellner/mapper/mapper.tar.gz

Since Python Mapper is not stable yet and under active development, the distribution will be updated frequently. If you want to follow updates more easily and avoid to install the same package over and over again, it is recommended to use our `Mercurial <http://mercurial.selenic.com/>`_ repository. Create a local copy of the repository with::

  hg clone http://math.stanford.edu/~muellner/hg/mapper

To update the repository, type::

  cd mapper
  hg pull
  hg up

Installation
------------

The Python Mapper archive can be extracted anywhere. Then add the directory where the files were extracted to `Python's search path <http://docs.python.org/2/install/#inst-search-path>`_. (Ie., add the directory which contains ``mapper`` as a subdirectory to the Python path.) The GUI tries to automatically adjust the Python path, so the last step is probably not necessary if you wish to use only the GUI.

(Later, I'll provide a proper ``distutils`` setup, so users will not need to worry about the Python path.)

Users may also want to add a link to the ``mapper/bin/MapperGUI.py`` script in a directory which is searched for executables. For example, my ``.bashrc`` contains a line::

  export PATH=${PATH+$PATH:}$HOME/.local/bin

so I can add a link to the GUI by::

  cd ~/.local/bin
  ln -s (MAPPER PATH)/bin/MapperGUI.py

Troubleshooting
---------------

If the GUI refuses to start with an error message like ::

  /usr/bin/env: python2: No such file or directory

there are three ways to deal with the problem:

* Do not call ``MapperGUI`` as an executable script but ``(your Python 2 interpreter) (your path)/MapperGUI.py``, eg.::

   python mapper/bin/MapperGUI.py

* Create a symbolic link like::

    sudo ln -s (path to the Python 2 interpreter) /usr/local/bin/python2

* Change the first line in ``MapperGUI.py`` from::

    #!/usr/bin/env python2

  to::

    #!/usr/bin/env (your Python 2 interpreter)

  With the last method, however, changes will be lost when Python Mapper is updated.

Compiling the documentation
---------------------------

This step is optional. The HTML documentation (this page!) can be compiled with `Sphinx <http://sphinx-doc.org/>`_::

  cd mapper/doc
  make html

If you get an error like ::

  make: sphinx-build2: No such file or directory

use::

  make html SPHINXBUILD=sphinx-build
