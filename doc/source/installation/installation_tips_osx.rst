Installation tips: OS X
=======================

I collect tips from other users here. Check them out and see whether they apply to your situation. Comments are welcome. Let me know if something becomes outdated.

.. rubric::  On July 16, 2013:

“Getting cmappertools installed properly in a virtual environment requires using the ``--prefix`` option. I'm managing my virtual environments with virtualenvwrapper, so all my envs live in ``.virtualenvs`` and I found that I had to ::

    python setup.py install --prefix=~/.virtualenvs/[MYENV]

where ``[MYENV]`` is the name of the environment.”

.. rubric::  On July 16, 2013:

“If people don't have wxPython installed, they need to be careful about how they go about installing it. If they try to ``import mapper`` and see the error ‘no module named wx’, they might try to ``pip install wx`` but this is bad because ``pip install wx`` doesn't install wxPython, but rather something else. I'm using `homebrew <http://brew.sh/>`_ to manage my packages (beneath `virtualenv <http://www.virtualenv.org>`_), and ``brew install wxwidgets`` will install what's needed. This still doesn't make wxPython work in a virtual environment though! I still have yet to sort that out.”

.. rubric::  On July 16, 2013:

“fastcluster can be installed with `pip <https://pypi.python.org/pypi/pip>`_, which some might prefer for its uninstall capability.”
