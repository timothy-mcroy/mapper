.. automodule:: mapper.filters
    :members:

    .. function:: eccentricity(data, exponent=1.0,  metricpar={}, callback=None)

        The eccentricity filter is the higher the further away a point is from the ‘center’ of the data. Note however that neither an explicit central point is required nor a barycenter in an embedding of the data. Instead, an intrinsic notion of centrality is derived from the pairwise distances.

        If *exponent* is finite:

        .. math::

          \mathit{eccentricity}(i)=\left(\frac 1N\sum_{j=0}^{N-1} d(x_i,x_j)^{\mathit{exponent}}\right)^{1/\mathit{exponent}}

        If *exponent* is ``numpy.inf``:

        .. math::

          \mathit{eccentricity}(i)=\max_j d(x_i,x_j)

        This is an equivalent description: Consider the full :math:`(N\times N)`-matrix of pairwise distances. The eccentricity of the :math:`i`-th data point is the Minkowski norm of the :math:`i`-th row with the respective *exponent*.

    .. function:: Gauss_density(data, sigma, metricpar={}, callback=None)

        Kernel density estimator with a multivariate, radially symmetric Gaussian kernel.

        For vector data and :math:`x\in\mathbb{R}^d`:

        .. math::

          \mathit{Gauss\_density}(x) = \frac{1}{N(\sqrt{2\pi}\sigma)^d} \sum_{j=0}^{N-1}\exp\left(-\frac{\|x-x_j\|^2}{2\sigma^2}\right)

        The density estimator is normalized to a probability measure, i.e. the integral of this function over :math:`\mathbb{R}^d` is 1. The  :math:`i`-th filter value is the density estimator evaluated at :math:`x_i`.

        For dissimilarity data:

        .. math::

          \mathit{Gauss\_density}(i) = \sum_{j=0}^{N-1}\exp\left(-\frac{d(x_i,x_j)^2}{2\sigma^2}\right)

        In this case, the density estimator is not normalized since there is no domain to integrate over.
