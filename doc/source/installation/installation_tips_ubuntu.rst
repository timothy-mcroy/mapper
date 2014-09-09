Installation tips: Ubuntu
=========================

The following sequence of commands in a terminal sets everything up in Ubuntu (tested with Ubuntu 13.10 “Saucy Salamander”)::

    sudo apt-get install python-numpy python-scipy python-matplotlib python-wxtools \
        python-opengl graphviz libboost-all-dev mercurial
    hg clone http://math.stanford.edu/~muellner/hg/mapper
    cd mapper/cmappertools
    python setup.py install --user
    cd ../..
    wget https://pypi.python.org/packages/source/f/fastcluster/fastcluster-1.1.13.tar.gz
    tar -xf fastcluster-1.1.13.tar.gz
    cd fastcluster-1.1.13
    python setup.py install --user
    cd ..
    rm fastcluster-1.1.13.tar.gz
    rm -r fastcluster-1.1.13/
