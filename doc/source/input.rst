Input data
==========

The input to Mapper is a ``numpy.ndarray`` with ``double`` data type. If it is two-dimensional (say of shape ``(m,n)`` with  *m* rows and *n* columns), the data is interpreted as vector data: *m* data points in :math:`\mathbb R^n`. If the array is one-dimensional, it is interpreted as pairwise distances. The array is the flattened, upper triangular part of the pairwise distance matrix. For *m* data points :math:`x_0,\ldots, x_{m-1}`, this vector contains :math:`\tbinom m 2=m(m-1)/2` entries. The first entriy is the distance :math:`d(x_0, x_1)`, the (*m*\ −1)-th entry is :math:`d(x_0, x_{m-1})`, the *m*\ -th entry is :math:`d(x_1, x_2)`, and so on.
