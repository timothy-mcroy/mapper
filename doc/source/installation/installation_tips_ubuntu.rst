Installation tips: Ubuntu
=========================

The following sequence of commands in a terminal sets everything up in Ubuntu (tested with Ubuntu 13.10 “Saucy Salamander”)::

    sudo apt-get install python-numpy python-scipy python-matplotlib python-wxtools \
        python-opengl graphviz libboost-all-dev mercurial
    hg clone http://math.stanford.edu/~muellner/hg/mapper
    cd mapper/cmappertools
    python setup.py install --user
    cd ../..
    wget http://cran.r-project.org/src/contrib/fastcluster_1.1.13.tar.gz
    tar -xf fastcluster_1.1.13.tar.gz
    cd fastcluster/src/python
    python setup.py install --user
    cd ../../..
    rm fastcluster_1.1.13.tar.gz
    rm -r fastcluster/
