What is Python Mapper?
======================

The Mapper algorithm is a method for topological data analysis invented by Gurjeet Singh, Facundo Mémoli and Gunnar Carlsson. See the Reference [R1]_ for the publication. While the Mapper algorithm alone does not constitute a complete data analysis tool itself, it is the key part of a processing chain with (minimally) filter functions, the Mapper algorithm itself and visualization of the results.

Python Mapper is a realization of this toolchain, written by Daniel Müllner and Aravindakshan Babu. It is open source software and is released under the GNU GPLv3 license.

There is also a company, `Ayasdi <http://www.ayasdi.com>`_, which was founded by Gurjeet Singh, Gunnar Carlsson and Harlan Sexton and whose main product, the `Ayasdi Iris software <http://www.ayasdi.com/product/>`_, has the Mapper algorithm at its core. Ayasdi also issues `academic trial licenses <http://www.ayasdi.com/inquiry/academic-trial.html>`_.

As much as Ayasdi covers the commercial uses of the Mapper algorithm with a polished and mature product, the authors of Python Mapper strive to provide the scientific community with a complete, extensible and fast toolchain. Since it is open source, it can be extended and modified by anyone with new ideas. The Python Mapper software also provides a graphical user interface, which hopefully makes it accessible to non-specialists and speeds up the workflow for beginners and experts alike.

To get an idea about the Mapper algorithm, we recommend Gunnar Carlsson's paper “Topology and Data” [R5]_ and the original Mapper paper [R1]_.
