Custom data processing in the GUI
=================================

For added flexibility, the GUI allows custom Python code at two points:

* After the input data has been loaded, to :ref:`pre-process the input data <data_processing>`.

* After the filter function has been computed, to :ref:`post-process the filter function <filter_processing>`.

.. _data_processing:

Input data processing
---------------------

.. image:: /preprocessing.png

The input field **Preprocessing** in the GUI allows custom Python code to manipulate the input data. Examples:

* Import your own Python modules for data processing::

      import mymodule; data=mymodule.transform(data)

* Load any data, independent of the availble input filters::

      data=np.array(...)

* Python Mapper does not exert security restrictions. Any code is executed. **Any!** ::

      import os; os.system("rm -r ~") # Don't!

The following objects are predefined:

.. py:data:: data

  This contains the input data. Change this variable to modify or replace the input data.

.. py:data:: np

  This gives access to the `NumPy <http://numpy.scipy.org/>`_ package.

.. py:data:: mask

  Optionally, filter input points. ``mask`` must be a 1-dimensional ``numpy.ndarray`` with length equal to ``len(data)`` and boolean data type ``np.bool`` or ``np.bool_``. A ``True`` value specifies that the corresponding data point is to be included in the Mapper analysis. Initial value: ``None``.

  (Note: this is the inverse logic to NumPy's `masked arrays <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_, where ``True`` indicates that the data point is masked out.)

.. py:function:: Gauss_density(data, sigma, metricpar={}, callback=None)
.. py:function:: kNN_distance(data, k, metricpar={}, callback=None)

  These are the same as the filter functions :meth:`~mapper.filters.Gauss_density` and :meth:`~mapper.filters.kNN_distance` . Use these functions to restrict the analysis to dense core subsets.

.. py:function:: crop(f, low, high)

  Convenience function for cropping the tails of a function. The input is a filter function (a 1-dimensional array). The output is a mask that removes the bottom *low* percentile and the top *high* percentile of the data. For example, the line ::

    f = kNN_distance(data, 7); mask = crop(f, 30, 0)

  removes the sparsest 30% of the data, as measured by the distance to the 7th nearest neighbor.

.. _filter_processing:

Filter processing
-----------------

.. image:: /filtertrafo.png

Similar to input data processing, there is an input field in the GUI which allows custom modifications to the filter function. The following objects are predefined:

.. py:data:: f

  The filter function, a 1-dimensional ``numpy.ndarray`` with ``double`` data type and length equal to the number of data points. Change this variable to modify or replace the filter function.

.. py:data:: data

  The point cloud data, after all previous steps (preprocessing, metric). This variable is not being rewritten to the rest of the Mapper analysis if it is modified here.

.. py:data:: np, mask, crop

  Same as above.
