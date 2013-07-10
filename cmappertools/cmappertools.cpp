/*
cmappertools: Tools for Python Mapper in C++

This file is part of the Python Mapper package, an open source tool
for exploration, analysis and visualization of data.

Copyright 2012–2013 Daniel Müllner, http://math.stanford.edu/~muellner

Python Mapper is distributed under the GPLv3 license. See the project home page

    http://math.stanford.edu/~muellner/mapper

for more information.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlong-long"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wformat"
#endif
#include <Python.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define BOOST_NO_HASH // Get rid of warning because of deprecated header
#include <boost/config.hpp>
#include <boost/thread.hpp>

#ifndef HAVE_VISIBILITY
#if __GNUC__ >= 4
#define HAVE_VISIBILITY 1
#endif
#endif

#if HAVE_VISIBILITY
#pragma GCC visibility push(hidden)
#endif

#include <boost/version.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggregate-return"
#endif
#include <boost/graph/connected_components.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#pragma GCC diagnostic ignored "-Waggregate-return"
#endif
//#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/dijkstra_shortest_paths_no_color_map.hpp>

#include <limits> // for infinity()
#include <new> // for bad_alloc
#include <string>
#include <algorithm> // for sort, partial_sort, copy
#include <numeric> // for partial_sum

#include "csr_graph.hpp" // my own version of compressed sparse row graphs

// For NumPy
// http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html#deprecation-mechanism-npy-no-deprecated-api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlong-long"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
#include <numpy/arrayobject.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <numpy/npy_math.h> // for NPY_INFINITY, NPY_SQRT2
#ifdef __GNUC__

#define CMT_Py_INCREF(X) _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    _Pragma("GCC diagnostic ignored \"-Wstrict-aliasing\"") \
    Py_INCREF(X); \
    _Pragma("GCC diagnostic pop")

#define CMT_Py_XINCREF(X) _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    Py_XINCREF(X); \
    _Pragma("GCC diagnostic pop")

#define CMT_Py_DECREF(X) _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    Py_DECREF(X); \
    _Pragma("GCC diagnostic pop")

#define CMT_Py_XDECREF(X) _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    Py_XDECREF(X); \
    _Pragma("GCC diagnostic pop")

#else

#define CMT_Py_INCREF(X) Py_INCREF(X)
#define CMT_Py_XINCREF(X) Py_XINCREF(X)
#define CMT_Py_DECREF(X) Py_DECREF(X)
#define CMT_Py_XDECREF(X) Py_XDECREF(X)

#endif

static PyObject*
CMT_PyArray_FROMANY(
    PyObject* obj,
    int typenum,
    int min,
    int max,
    int requirements)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
  return PyArray_FROMANY(obj, typenum, min, max, requirements);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

static inline int
CMT_PyDict_Check(PyObject* p)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
  return PyDict_Check(p);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

static inline PyObject*
CMT_PyArray_ZEROS(int nd, npy_intp* dims, int type_num, int fortran)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
  return PyArray_ZEROS(nd, dims, type_num, fortran);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

static inline PyObject*
CMT_PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
  return PyArray_SimpleNew(nd, dims, typenum);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

static inline npy_intp
CMT_PyArray_SIZE(PyArrayObject* arr)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
  return PyArray_SIZE(arr);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

static inline void*
CMT_PyArray_GETPTR2(PyArrayObject* obj, npy_intp i, npy_intp j)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
  return PyArray_GETPTR2(obj, i, j);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

// backwards compatibility with the old NumPy C API
#ifndef NPY_ARRAY_CARRAY_RO
#define NPY_ARRAY_CARRAY_RO NPY_CARRAY_RO
#endif
#ifndef NPY_ARRAY_CARRAY
#define NPY_ARRAY_CARRAY NPY_CARRAY
#endif
#ifndef NPY_ARRAY_IN_ARRAY
#define NPY_ARRAY_IN_ARRAY NPY_IN_ARRAY
#endif
#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

#include "config.h"

bool const multithreading = true;

static char const __version__[] = PACKAGE_VERSION;

///////////////////////////////////////////////////////////////////////////////

// Self-destructing array pointer, similar to boost::scoped_array.
// The allocated memory is freed when the object is destroyed.
template<typename type>
  class auto_array_ptr
  {
  private:
    type * ptr;

    // Noncopyable
    auto_array_ptr(auto_array_ptr const &);
    auto_array_ptr &
    operator=(auto_array_ptr const &);

  public:
    auto_array_ptr()
        : ptr(NULL)
    {
    }
    template<typename index>
      explicit
      auto_array_ptr(index const size)
          : ptr(new type[size])
      {
      }
    template<typename index, typename value>
      auto_array_ptr(index const size, value const val)
          : ptr(new type[size])
      {
        std::fill_n(ptr, size, val);
      }
    ~auto_array_ptr()
    {
      delete[] ptr;
    }
    void
    free()
    {
      delete[] ptr;
      ptr = NULL;
    }
    template<typename index>
      void
      init(index const size)
      {
        ptr = new type[size];
      }
    template<typename index, typename value>
      void
      init(index const size, value const val)
      {
        init(size);
        std::fill_n(ptr, size, val);
      }
    inline
    operator type *() const
    {
      return ptr;
    }
  };

///////////////////////////////////////////////////////////////////////////////

struct errormessage
{
  PyObject * const errortype;
  char const * const message;
};

static errormessage const err_inv_metric =
  { PyExc_ValueError, "Invalid 'metric' parameter." };
static errormessage const err_wrong_stride =
  { PyExc_ValueError, "Unexpected stride for dissimilarity array." };
static errormessage const err_no_dm =
      { PyExc_ValueError,
          "The number of data elements does not correspond to a compressed distance matrix." };
static errormessage const err_metric_not_allowed =
  { PyExc_ValueError,
      "No optional parameter is allowed for a dissimilarity matrix." };
static errormessage const err_no_dict =
  { PyExc_TypeError, "The 'metricpar' parameter must be a dictionary." };
static errormessage const err_callback_not_callable =
  { PyExc_TypeError, "The 'callback' parameter must be callable." };
static errormessage const err_dm_stride =
  { PyExc_ValueError, "The dissimilarity array must be aligned." };
static errormessage const err_toomanypoints =
  { PyExc_IndexError, "Too many input points for the Python 'int' data type." };
static errormessage const err_k =
  { PyExc_ValueError,
      "The parameter k must be between 1 and N, the number of data points." };
static errormessage const err_eps =
  { PyExc_ValueError, "The parameter eps must be nonnegative." };
static errormessage const err_sigma_eps =
  { PyExc_ValueError, "The parameter sigma_eps must be positive." };
static errormessage const err_n =
  { PyExc_ValueError,
      "The index array must not be bigger than the number of data points." };
static errormessage const err_num_clust =
      { PyExc_ValueError,
          "The number of clusters must between 1 and the number of points in the dendrogram." };
static errormessage const err_rowstartzero =
  { PyExc_ValueError, "The 'rowstart' vector has length zero." };
static errormessage const err_samesize =
  { PyExc_ValueError,
      "The 'targets' and 'weights' vectors must have the same size." };
static errormessage const err_noloopedge =
  { PyExc_ValueError,
      "The first edge of each vertex must be the loop edge with weight zero." };

///////////////////////////////////////////////////////////////////////////////

static npy_intp
n_obs(npy_intp NN)
{
  npy_intp N;
  if (NN == 0)
  {
    N = 1;
  }
  else
  {
    N = static_cast<npy_intp>(ceil(
        sqrt(static_cast<npy_double>(NN)) * NPY_SQRT2));
    if (NN != N * (N - 1) >> 1)
      throw(err_no_dm);
  }
  return N;
}

///////////////////////////////////////////////////////////////////////////////

// Class for pairwise distances and kernels

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#endif
class metric_and_kernel
{
  typedef void
  (metric_and_kernel::*metricfnptr)(npy_intp const i);
private:
  npy_double p;
  char const * const X;
  npy_intp const * const dims;
  npy_intp const * const strides;
  auto_array_ptr<npy_double> out;
  npy_double * out_offset;
  npy_double exponent_to_do;
  metricfnptr cdistfn;
  bool const read_only_data;

  static char const * const str_euclidean;
  static char const * const str_minkowski;
  static char const * const str_chebychev;

// Noncopyable
  metric_and_kernel(metric_and_kernel const &);
  metric_and_kernel &
  operator=(metric_and_kernel const &);

public:
// Constructor for vector data - involves the choice of a metric
  metric_and_kernel(
      PyObject * const metric_obj,
      npy_double const p_,
      char const * const X_,
      npy_intp const * const dims_,
      npy_intp const * const strides_)
      : p(p_),
        X(X_),
        dims(dims_),
        strides(strides_),
        out(dims[0]),
        out_offset(NULL), // unnecessary, suppress compiler warning
        exponent_to_do(1),
        cdistfn(&metric_and_kernel::chebychev),
        read_only_data(false)
  {
    if (!metric_obj || compare_as_unicode(metric_obj, str_euclidean))
    {
      cdistfn = &metric_and_kernel::euclidean;
      exponent_to_do = .5;
    }
    else if (compare_as_unicode(metric_obj, str_minkowski))
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
      if (p != std::numeric_limits<npy_double>::infinity())
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
      {
        cdistfn = &metric_and_kernel::minkowski;
        exponent_to_do = 1 / p;
      }
    }
    else if (!compare_as_unicode(metric_obj, str_chebychev))
    {
      throw(err_inv_metric);
    }
  }

// Constructor for dissimilarity data - no metric needed
  metric_and_kernel(
      char const * const X_,
      npy_intp const * const strides_,
      npy_intp const * const N)
      : p(), // unnecessary, suppress compiler warning
        X(X_),
        dims(N),
        strides(strides_),
        out(),
        out_offset(NULL), // unnecessary, suppress compiler warning
        exponent_to_do(1),
        cdistfn(&metric_and_kernel::no_op), //&metric_and_kernel::dm;
        read_only_data(true)
  {
    if (strides[0] != sizeof(npy_double))
    {
      throw(err_wrong_stride);
    }
  }

  inline void
  cdist(npy_intp const i, npy_intp offset = 0)
  {
    out_offset = out - offset;
    (this->*cdistfn)(i);
  }

  npy_double const *
  pow_p(npy_double const exp, npy_intp const i, npy_intp const offset)
  {
    npy_double const * in;
    npy_double const final_exponent = exponent_to_do * exp;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    if (final_exponent == 1)
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    {  // do nothing
      if (read_only_data)
      {
        return dm(i);
      }
      else
      {
        out_offset = out - offset;
        return out_offset;
      }
    }
    if (read_only_data)
    {
      if (!&*out)
      {
        out.init(dims[0]);
      }
      out_offset = out - offset;
      in = dm(i);
    }
    else
    {
      out_offset = out - offset;
      in = out_offset;
    }
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    if (final_exponent == .5)
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    { // square root
      for (npy_intp j = i + 1; j < dims[0]; ++j)
      {
        out_offset[j] = sqrt(in[j]);
      }
    }
    else
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
      if (final_exponent == 2)
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
      { // square
        for (npy_intp j = i + 1; j < dims[0]; ++j)
        {
          out_offset[j] = in[j] * in[j];
        }
      }
      else
      { // all other exponents
        for (npy_intp j = i + 1; j < dims[0]; ++j)
        {
          out_offset[j] = pow(in[j], final_exponent);
        }
      }
    }
    return out_offset;
  }

  npy_double const *
  Gauss_kernel(npy_double const v, npy_intp const i, npy_intp const offset)
  {
    npy_double const * in = pow_p(2, i, offset);
    if (!&*out)
    {
      out.init(dims[0]);
      out_offset = out - offset;
    }
    for (npy_intp j = i + 1; j < dims[0]; ++j)
    {
      out_offset[j] = exp(in[j] / v);
    }
    return out_offset;
  }

private:
  bool
  compare_as_unicode(PyObject * s1, char const * const s2)
  {
    PyObject * s2u = PyUnicode_FromString(s2);
    if (!s2u)
    {
      throw err_inv_metric;
    }
    bool ret = PyUnicode_Compare(s1, s2u) == 0;
    CMT_Py_DECREF(s2u);
    return ret;
  }

  void
  euclidean(npy_intp const i)
  {
    char const * const a = X + strides[0] * i;
    for (npy_intp j = i + 1; j < dims[0]; ++j)
    {
      out_offset[j] = 0;
      for (npy_intp k = 0; k < dims[1]; ++k)
      {
        npy_double const diff = *reinterpret_cast<npy_double const *>(a
            + strides[1] * k)
            - *reinterpret_cast<npy_double const *>(X + strides[0] * j
                + strides[1] * k);
        out_offset[j] += diff * diff;
      }
    }
  }

  void
  minkowski(npy_intp const i)
  {
    char const * const a = X + strides[0] * i;
    for (npy_intp j = i + 1; j < dims[0]; ++j)
    {
      out_offset[j] = 0;
      for (npy_intp k = 0; k < dims[1]; ++k)
      {
        out_offset[j] += pow(
            fabs(
                *reinterpret_cast<npy_double const *>(a + strides[1] * k)
                    - *reinterpret_cast<npy_double const *>(X + strides[0] * j
                        + strides[1] * k)), p);
      }
    }
  }

  void
  chebychev(npy_intp const i)
  {
    char const * const a = X + strides[0] * i;
    for (npy_intp j = i + 1; j < dims[0]; ++j)
    {
      out_offset[j] = 0;
      for (npy_intp k = 0; k < dims[1]; ++k)
      {
        npy_double const d = fabs(
            *reinterpret_cast<npy_double const *>(a + strides[1] * k)
                - *reinterpret_cast<npy_double const *>(X + strides[0] * j
                    + strides[1] * k));
        if (d > out_offset[j])
        {
          out_offset[j] = d;
        }
      }
    }
  }

  inline void
  no_op(npy_intp const)
  {
  }

  inline npy_double const *
  dm(npy_intp const i)
  {
    return reinterpret_cast<npy_double const *>(X)
        + ((2 * dims[0] - 3 - i) * i >> 1) - 1;
  }
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

char const * const metric_and_kernel::str_euclidean = "euclidean";
char const * const metric_and_kernel::str_minkowski = "minkowski";
char const * const metric_and_kernel::str_chebychev = "chebychev";

///////////////////////////////////////////////////////////////////////////////

// Classes for kernel functions (both eccentricity and Gaussian density estimator)

class kernel_func_class
{
protected:
  npy_double * const ecc;
  npy_intp const n;

private:
// Noncopyable
  kernel_func_class(kernel_func_class const &);
  kernel_func_class &
  operator=(kernel_func_class const &);

public:
  kernel_func_class(npy_double * const ecc_, npy_intp const n_)
      : ecc(ecc_),
        n(n_)
  {
  }

  virtual
  ~kernel_func_class()
  {
  } //  Virtual destructor does nothing
// but prevents a warning (-Wdelete-non-virtual-dtor for the GCC).

  virtual npy_double const *
  preprocess(
      metric_and_kernel * const m,
      npy_intp const i,
      npy_intp const offset = 0)
  {
    return m->pow_p(1, i, offset);
  }
  virtual void
  process(npy_double const * const d, npy_intp const i, npy_double * const out)
  {
    for (npy_intp j = i + 1; j < n; ++j)
    {
      out[i] += d[j];
      out[j] += d[j];
    }
  }
  inline void
  process(npy_double const * const d, npy_intp const i)
  {
    this->process(d, i, ecc);
  }
  virtual void
  mergeresult(npy_double * const out)
  {
    for (npy_intp i = 0; i < n; ++i)
    {
      ecc[i] += out[i];
    }
  }
  virtual void
  postprocess()
  {
  }
};

class l1ecc : public kernel_func_class
{
public:
  l1ecc(npy_double * const ecc_, npy_intp const n_)
      : kernel_func_class(ecc_, n_)
  {
  }

  virtual void
  postprocess()
  {
    npy_double const n_double = static_cast<npy_double>(n);
    for (npy_intp i = 0; i < n; ++i)
    {
      ecc[i] /= n_double;
    }
  }
};

class lpecc : public kernel_func_class
{
  npy_double const exponent;

public:
  lpecc(npy_double * const ecc_, npy_intp const n_, npy_double const exponent_)
      : kernel_func_class(ecc_, n_),
        exponent(exponent_)
  {
  }

  virtual npy_double const *
  preprocess(
      metric_and_kernel * const m,
      npy_intp const i,
      npy_intp const offset = 0)
  {
    return m->pow_p(exponent, i, offset);
  }
  virtual void
  postprocess()
  {
    npy_double const n_double = static_cast<npy_double>(n);
    npy_double const expinv = 1 / exponent;
    for (npy_intp i = 0; i < n; ++i)
    {
      ecc[i] = pow(ecc[i] / n_double, expinv);
    }
  }
};

class linfecc : public kernel_func_class
{
public:
  linfecc(npy_double * const ecc_, npy_intp const n_)
      : kernel_func_class(ecc_, n_)
  {
  }

  virtual void
  process(npy_double const * const d, npy_intp const i, npy_double * const out)
  {
    for (npy_intp j = i + 1; j < n; ++j)
    {
      if (d[j] > out[i])
        out[i] = d[j];
      if (d[j] > out[j])
        out[j] = d[j];
    }
  }
  virtual void
  mergeresult(npy_double * const out)
  {
    for (npy_intp i = 0; i < n; ++i)
    {
      if (out[i] > ecc[i])
        ecc[i] = out[i];
    }
  }
};

class Gauss_kernel : public kernel_func_class
{
  npy_double const v;
  npy_double const denom;
public:
  Gauss_kernel(
      npy_double * const ecc_,
      npy_intp const n_,
      npy_double const v_,
      npy_double const denom_)
      : kernel_func_class(ecc_, n_),
        v(v_),
        denom(denom_)
  {
  }

  virtual npy_double const *
  preprocess(
      metric_and_kernel * const m,
      npy_intp const i,
      npy_intp const offset = 0)
  {
    return m->Gauss_kernel(v, i, offset);
  }
  virtual void
  postprocess()
  {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    if (denom != npy_double(1))
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    {
      for (npy_intp i = 0; i < n; ++i)
      {
        ecc[i] = (ecc[i] + 1) / denom;
      }
    }
    else
    {
      for (npy_intp i = 0; i < n; ++i)
      {
        ecc[i] += 1;
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

// Progress reporter

static char format_int[] = "i";

class Callback
{
private:
  PyThreadState * * const PythonThreadSave;
  PyObject * const callback;
  npy_intp oldpercent;

// Noncopyable
  Callback(Callback const &);
  Callback &
  operator=(Callback const &);

public:
  Callback(PyThreadState * * PythonThreadSave_, PyObject * callback_)
      : PythonThreadSave(PythonThreadSave_),
        callback(callback_),
        oldpercent(-1)
  {
  }

  void
  operator()(npy_intp percent)
  {
    if (!callback)
      return;
    if (percent > 100)
      percent = 100;
    if (percent != oldpercent)
    {
      oldpercent = percent;
      if (*PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(*PythonThreadSave);
      }
      PyObject * answer = PyObject_CallFunction(callback, format_int, percent);
      CMT_Py_XDECREF(answer);
      *PythonThreadSave = PyEval_SaveThread();
    }
  }

  void
  RestoreAndSend100Percent()
  {
    if (*PythonThreadSave)
    { // Only restore if the state has been saved
      PyEval_RestoreThread(*PythonThreadSave);
    }
    if (callback && 100 != oldpercent)
    {
      PyObject * answer = PyObject_CallFunction(callback, format_int, 100);
      CMT_Py_XDECREF(answer);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

static char arg_s1[] = "data";
static char arg_s2a[] = "exponent";
static char arg_s2b[] = "sigma";
static char arg_s3[] = "metricpar";
static char arg_s4[] = "callback";
static char arg_s5[] = "verbose";
static char arg_t1[] = "rowstart";
static char arg_t2[] = "targets";
static char arg_t3[] = "weights";
static char arg_t4[] = "D";
static char arg_t5[] = "k";
static char arg_t6[] = "eps";
static char arg_t7[] = "diagonal";

namespace kernel_filter
{

  enum kernel_method
  {
    method_eccentricity, method_Gauss_density
  };

  static PyObject *
  common_kernel_filter(
      kernel_method const method,
      PyObject * const args,
      PyObject * const kwargs);

  inline static PyObject *
  eccentricity(PyObject * const, // self, not needed
      PyObject * const args,
      PyObject * const kwargs)
  {
    return common_kernel_filter(method_eccentricity, args, kwargs);
  }

  inline static PyObject *
  Gauss_density(PyObject * const, // self, not needed
      PyObject * const args,
      PyObject * const kwargs)
  {
    return common_kernel_filter(method_Gauss_density, args, kwargs);
  }

// processrow is called as separate threads for parallel processing

  static void
  processrow(
      boost::exception_ptr & error,
      npy_intp i,
      npy_intp * const iptr,
      boost::mutex * mutex_i,
      boost::mutex * mutex_result,
      npy_intp const n,
      metric_and_kernel * const m,
      kernel_func_class * kernel_func)
  {
    try
    {
      npy_intp const stopi = (n - 1) / 2;
      auto_array_ptr<npy_double> rt(n, 0);
      while (i < stopi)
      {
        npy_intp const i2 = n - 2 - i;
        m->cdist(i, i + 1);
        m->cdist(i2, 0);
        npy_double const * const d = kernel_func->preprocess(m, i, i + 1);
        npy_double const * const d2 = kernel_func->preprocess(m, i2, 0);
        kernel_func->process(d, i, rt);
        kernel_func->process(d2, i2, rt);

        mutex_i->lock();
        i = *iptr;
        ++(*iptr);
        mutex_i->unlock();
      }
      if (2 * i == n - 2)
      {
        m->cdist(i);
        npy_double const * const d = kernel_func->preprocess(m, i);
        kernel_func->process(d, i, rt);
      }
      mutex_result->lock();
      kernel_func->mergeresult(rt);
      mutex_result->unlock();
      error = boost::exception_ptr(); // no error
    }
    catch (std::exception & e)
    {
      error = boost::current_exception();
    }
  }

  static PyObject *
  common_kernel_filter(
      kernel_method const method,
      PyObject * const args,
      PyObject * const kwargs)
  {
    PyObject * data;
    npy_double exponent_sigma = 1;
    // exponent _or_ sigma, depending on the method
    PyObject * metricpar = NULL;
    npy_double p = 2;

    PyThreadState * PythonThreadSave = NULL;
    PyObject * callback = NULL;

    static char * kwlist[] =
      { arg_s1, NULL, arg_s3, arg_s4, NULL };
    kwlist[1] = method == method_eccentricity ? arg_s2a : arg_s2b;

    PyArrayObject * data_double = NULL;

    PyArrayObject * rslt_npy = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dOO", kwlist, &data,
          &exponent_sigma, &metricpar, &callback))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (!(data_double = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          data, NPY_DOUBLE, 1, 2, NPY_ARRAY_ALIGNED))))
      {
        return NULL;
      }
      char const * const X = reinterpret_cast<char const *>(PyArray_DATA(
          data_double));

      PyObject* metric_obj = NULL;
      if (metricpar)
      {
        if (!CMT_PyDict_Check(metricpar))
        {
          throw(err_no_dict);
        }
        metric_obj = PyDict_GetItemString(metricpar, "metric");
        PyObject* p_obj = PyDict_GetItemString(metricpar, "p");
        if (p_obj)
        {
          p = PyFloat_AsDouble(p_obj);
          if (PyErr_Occurred())
          {
            return NULL;
          }
        }
      }

      if (callback == Py_None)
      {
        callback = NULL;
      }
      else if (callback && !PyCallable_Check(callback))
      {
        throw(err_callback_not_callable);
      }
      Callback SendProgress(&PythonThreadSave, callback);

      npy_intp const ndim = PyArray_NDIM(data_double);

      unsigned int nthreads =
          multithreading ? boost::thread::hardware_concurrency() : 1;

      std::vector<metric_and_kernel *> mk(nthreads);

      npy_intp const * const dims = PyArray_DIMS(data_double);
      npy_intp const * const strides = PyArray_STRIDES(data_double);
      npy_intp n;
      if (ndim == 1)
      {
        if (strides[0] != sizeof(npy_double))
          throw(err_dm_stride);
        n = n_obs(dims[0]);
        if (metric_obj)
        { //fixme
          throw(err_metric_not_allowed);
        }
        mk[0] = new metric_and_kernel(X, PyArray_STRIDES(data_double), &n);
        // Parallel computation does not save time for a distance matrix.
        nthreads = 1;
      }
      else
      { // ndim must be 2
        n = dims[0];
        if (n < nthreads)
        {
          nthreads = 1;
        }
        for (unsigned int i = 0; i < nthreads; ++i)
        {
          mk[i] = new metric_and_kernel(metric_obj, p, X, dims, strides);
        }
      }

      if (!(rslt_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_ZEROS(1,
          &n, NPY_DOUBLE, false))))
      {
        return NULL;
      }
      // false ⇒ C order
      npy_double * const ecc = reinterpret_cast<npy_double *>(PyArray_DATA(
          rslt_npy));

      PythonThreadSave = PyEval_SaveThread();

      kernel_func_class * kernel_func;
      if (method == method_eccentricity)
      {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        if (exponent_sigma == 1)
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
        {
          kernel_func = new l1ecc(ecc, n);
        }
        else
        {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
          if (exponent_sigma == std::numeric_limits<npy_double>::infinity())
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
          {
            kernel_func = new linfecc(ecc, n);
          }
          else
          {
            kernel_func = new lpecc(ecc, n, exponent_sigma);
          }
        }
      }
      else
      {
        /* If we know the dimensionality, we can normalize the density
         estimator to a probability measure. With unknown dimensionality
         (dissimilarity input, we do not normalize). */
        kernel_func = new Gauss_kernel(ecc, n,
            -2 * exponent_sigma * exponent_sigma,
            ndim == 1 ?
                1 :
                static_cast<npy_double>(n)
                    * pow(sqrt(2 * M_PI) * exponent_sigma,
                        static_cast<npy_double>(dims[1])));
      }

      //printf("Number of threads: %d\n", nthreads);
      if (nthreads == 1)
      {
        for (npy_intp i = 0; i < n; ++i)
        {
          mk[0]->cdist(i);
          npy_double const * const d = kernel_func->preprocess(mk[0], i);
          kernel_func->process(d, i);
          SendProgress(i * 100 / n);
        }
      }
      else
      {
        boost::mutex mutex_i, mutex_result;
        std::vector<boost::thread *> threads(nthreads);
        npy_intp row = nthreads;
        std::vector<boost::exception_ptr> error(nthreads);
        for (unsigned int i = 0; i < nthreads; ++i)
        {
          threads[i] = new boost::thread(processrow, boost::ref(error[i]), i,
              &row, &mutex_i, &mutex_result, n, mk[i], kernel_func);
        }
        while (
#if BOOST_VERSION < 105000
        !threads[0]->timed_join(boost::posix_time::milliseconds(200))
#else
        !threads[0]->try_join_for(boost::chrono::milliseconds(200))
#endif
        )
        {
          SendProgress(row * 200 / n);
        }
        for (unsigned int i = 0; i < nthreads; ++i)
        {
          if (i > 0)
            threads[i]->join();
          if (error[i])
          {
            boost::rethrow_exception(error[i]);
          }
          delete threads[i];
          delete mk[i];
        }
      }
      kernel_func->postprocess();
      delete kernel_func;

      SendProgress.RestoreAndSend100Percent();
      CMT_Py_DECREF(data_double);
      return reinterpret_cast<PyObject *>(rslt_npy);
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(data_double);
      CMT_Py_XDECREF(rslt_npy);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(data_double);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(data_double);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(data_double);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

namespace graph
{

  typedef npy_intp Vertex; // see below: the data types used in Py_FROMANY
// This implementation needs a signed integer type. See "if (L<=0)" below.
  typedef npy_intp EdgeIndex;
  typedef npy_double Weight;
  typedef csr_graph::csr_graph<Vertex, EdgeIndex, Weight> Graph;

  /* Class for generating a CSR graph (CSR = compressed sparse row format)

   This class requires a two-pass process for generation. In the first pass,
   edges are registered in order to allocate the correct amount of memory.
   In the second pass, the edges are actually added.

   The resulting graph is weighted and bidirectional, ie every edge has a
   weight, and it appears twice in the list of edges: as (a,b) and (b,a)
   for both directions.

   The required workflow to generate a graph is as follows:

   G = csr_graph_generator( [number of vertices] );
   G.allocate_edge( a1, b1 ); // two vertices of the edge
   G.allocate_edge( a2, b2 );
   ...
   G.finalize();
   G.add_egde( a1, b1, t4 ); //  vertices and edge weight
   G.add_egde( a2, b2, t6 );
   ...

   It is important that exactly the same edges are added as were allocated
   (although the order does not matter).

   After this, the graph contains three pointers to NumPy arrays:

   rowstart_npy (length: number of vertices + 1)
   targets_npy  (length: number of edges)
   weights_npy  (length: number of edges)

   These 1-dimensional arrays contain the graph structure. "targets" is the
   list of all edge targets, and "weights" is the list of all edge weights.
   The edge sources are given implicitly: edges rowstart[0] to rowstart[1]-1
   start from vertex 0, edges rowstart[1] to rowstart[2]-1 start from vertex
   1. and so on. The entry rowstart[0] is always zero, and rowstart[-1] is
   the total number of edges.

   The NumPy arrays are generated freshly and should therefore have reference
   count 1. Don't forget to decrease the reference count if the arrays are
   no longer needed.
   */

  class csr_graph_generator
  {
  private:
    Vertex const N;
    EdgeIndex * rowstart_1;
    Vertex * targets;
    Weight * weights;

    // Noncopyable
    csr_graph_generator(csr_graph_generator const &);
    csr_graph_generator &
    operator=(csr_graph_generator const &);

  public:
    PyArrayObject * rowstart_npy, *targets_npy, *weights_npy;

  public:
    csr_graph_generator(Vertex const N_)
        : N(N_),
          rowstart_1(), // unnecessary, suppress compiler warning
          targets(), // unnecessary, suppress compiler warning
          weights(), // unnecessary, suppress compiler warning
          rowstart_npy(), // unnecessary, suppress compiler warning
          targets_npy(), // unnecessary, suppress compiler warning
          weights_npy() // unnecessary, suppress compiler warning

    {
      Vertex NpO = N + 1;
      rowstart_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_ZEROS(1,
          &NpO,
          NPY_INTP, 0));
      rowstart_1 = reinterpret_cast<EdgeIndex *>(PyArray_DATA(rowstart_npy))
          + 1;
    }

    void
    allocate_edge(Vertex const a, Vertex const b) const
    {
      ++rowstart_1[a];
      ++rowstart_1[b];
    }

    EdgeIndex
    finalize(PyThreadState * * const PythonThreadSave, bool diagonal = false)
    {
      EdgeIndex next;
      EdgeIndex num_edges = 0;
      for (EdgeIndex * r = rowstart_1; r != rowstart_1 + N; ++r)
      {
        next = *r + diagonal;
        *r = num_edges;
        num_edges += next;
      }

      PyEval_RestoreThread(*PythonThreadSave);

      targets_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(1,
          &num_edges, NPY_INTP));
      targets = reinterpret_cast<Vertex *>(PyArray_DATA(targets_npy));

      weights_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(1,
          &num_edges, NPY_DOUBLE));
      weights = reinterpret_cast<Weight *>(PyArray_DATA(weights_npy));

      *PythonThreadSave = PyEval_SaveThread();

      // If the diagonal is included in the graph, add the loop edges as the
      // first edge of each vertex.
      if (diagonal)
      {
        for (Vertex a = 0; a < N; ++a)
        {
          EdgeIndex const i = rowstart_1[a]++;
          targets[i] = a;
          weights[i] = 0;
        }
      }
      return num_edges;
    }

    void
    add_edge(Vertex const a, Vertex const b, Weight const w) const
    {
      EdgeIndex i;
      i = rowstart_1[a]++;
      targets[i] = b;
      weights[i] = w;
      i = rowstart_1[b]++;
      targets[i] = a;
      weights[i] = w;
    }

  };

  struct Vertex_Weight
  {
    /* Pair of (vertex, weight). Sorting is done according to edge weight. */
    Vertex v;
    Weight const * w;

    bool
    operator<(const Vertex_Weight& b) const
    {
      return (*w) < (*(b.w));
    }
  };

  struct Edge_ID
  {
    Vertex a, b;
    Weight const * id;

    inline bool
    operator<(const Edge_ID& other) const
    {
      /* Yes, we compare pointers, not values. The comparison operator is
       for a sorting process to filter out unique edges. We identify
       each edge uniquely by a pointer to the entry in the big dissimilarity
       array where the weight is recorded.
       */
      return id < other.id;
    }
    inline bool
    operator==(const Edge_ID& other) const
    {
      return id == other.id;
    }
    inline bool
    operator!=(const Edge_ID& other) const
    {
      return id != other.id;
    }
  };

  static PyObject *
  neighborhood_graph(PyObject * const, // self, not needed
      PyObject * const args,
      PyObject * const kwargs)
  {
    PyObject * D_py;
    PyArrayObject * D_npy = NULL;

    PyThreadState * PythonThreadSave = NULL;
    Py_ssize_t k = 1;
    double eps = 0;
    PyObject * diagonal_py = NULL;
    PyObject * callback = NULL;
    PyObject * verbose_py = NULL;

    static char * kwlist[] =
      { arg_t4, arg_t5, arg_t6, arg_t7, arg_s4, arg_s5, NULL }; // "D", "k",
    // "eps", "diagonal",
    // "callback", "verbose"

    PyObject * rslt_py = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ndOOO", kwlist, &D_py,
          &k, &eps, &diagonal_py, &callback, &verbose_py))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (callback == Py_None)
      {
        callback = NULL;
      }
      else if (callback && !PyCallable_Check(callback))
      {
        throw(err_callback_not_callable);
      }
      Callback SendProgress(&PythonThreadSave, callback);

      int diagonal = false;
      if (diagonal_py)
      {
        diagonal = PyObject_IsTrue(diagonal_py);
        if (diagonal == -1)
          return NULL;
      }

      int verbose = true;
      if (verbose_py)
      {
        verbose = PyObject_IsTrue(verbose_py);
        if (verbose == -1)
          return NULL;
      }

      if (!(D_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(D_py,
          NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      Weight const * const D = reinterpret_cast<Weight const *>(PyArray_DATA(
          D_npy));

      npy_intp const N = n_obs(CMT_PyArray_SIZE(D_npy));

      if (k < 1 || k > N)
        throw(err_k);
      if (eps < 0)
        throw(err_eps);

      PythonThreadSave = PyEval_SaveThread();

      csr_graph_generator G(N);

      Weight const * Dptr = D;
      if (verbose)
      {
        auto_array_ptr<Weight> min_eps_arr(N,
            std::numeric_limits<Weight>::infinity());

        for (Vertex ii = 0; ii < N; ++ii)
        {
          for (Vertex jj = ii + 1; jj < N; ++jj, ++Dptr)
          {
            if (*Dptr < min_eps_arr[ii])
            {
              min_eps_arr[ii] = *Dptr;
            }
            if (*Dptr < min_eps_arr[jj])
            {
              min_eps_arr[jj] = *Dptr;
            }
            if (*Dptr <= eps)
            {
              G.allocate_edge(ii, jj);
            }
          }
        }

        PyEval_RestoreThread(PythonThreadSave);
        PySys_WriteStdout("Minimal epsilon: %g.\n",
            *std::max_element(&*min_eps_arr, min_eps_arr + N));
        if (PyErr_Occurred())
        {
          return NULL;
        }
        PythonThreadSave = PyEval_SaveThread();
      }
      else
      {
        for (Vertex ii = 0; ii < N; ++ii)
        {
          for (Vertex jj = ii + 1; jj < N; ++jj, ++Dptr)
          {
            if (*Dptr <= eps)
            {
              G.allocate_edge(ii, jj);
            }
          }
        }
      }

      /* After the first batch of edge allocations, the vertex degrees
       are recorded in the "rowstart" array.
       */
      auto_array_ptr<EdgeIndex> degrees(N);
      EdgeIndex const * const rowstart_1 =
          reinterpret_cast<EdgeIndex *>(PyArray_DATA(G.rowstart_npy)) + 1;
      std::copy(rowstart_1, rowstart_1 + N, &*degrees);

      auto_array_ptr<Vertex_Weight> vw(N);
      auto_array_ptr<Edge_ID> ew(N * (k - 1));
      Vertex l = 0;
      Edge_ID * EW = &*ew;

      for (Vertex ii = 0; ii < N; ++ii)
      {
        Vertex const L = k - 1 - degrees[ii];
        if (L <= 0)
          continue; // We found already at least k-1 neighbors.
        l = 0;
        Dptr = D + ii - 1;
        for (Vertex jj = 0; jj < ii; ++jj)
        {
          if (*Dptr > eps)
          {
            vw[l].v = jj;
            vw[l++].w = Dptr;
          }
          Dptr += N - jj - 2;
        }
        for (Vertex jj = ii + 1; jj < N; ++jj)
        {
          ++Dptr;
          if (*Dptr > eps)
          {
            vw[l].v = jj;
            vw[l++].w = Dptr;
          }
        }
        if (l > L)
          std::partial_sort(&*vw, vw + L, vw + l);

        for (Vertex_Weight * VW = vw; VW != vw + std::min(l, L); ++VW)
        {
          EW->a = ii;
          EW->b = VW->v;
          EW->id = VW->w;
          ++EW;
        }
        SendProgress(ii * 100 / N);
      }

      degrees.free();
      vw.free();

      // Filter out duplicates
      std::sort(&*ew, EW);
      {
        Weight const * prev = NULL;
        for (Edge_ID const * e = &*ew; e != EW; ++e)
        {
          if (prev != e->id)
          {
            prev = e->id;
            G.allocate_edge(e->a, e->b);
          }
        }
      }

      EdgeIndex const num_edges = G.finalize(&PythonThreadSave, diagonal);

      if (verbose)
      {
        PyEval_RestoreThread(PythonThreadSave);
        PySys_WriteStdout("Density of the adjacency matrix: %.2f%%.\n",
            (static_cast<double>(num_edges - N * diagonal) * 100
                / static_cast<double>(N * (N - 1))));
        if (PyErr_Occurred())
        {
          return NULL;
        }
        PythonThreadSave = PyEval_SaveThread();
      }

      Dptr = D;
      for (Vertex ii = 0; ii < N; ++ii)
      {
        for (Vertex jj = ii + 1; jj < N; ++jj, ++Dptr)
        {
          if (*Dptr <= eps)
          {
            G.add_edge(ii, jj, *Dptr);
          }
        }
      }

      {
        Weight const * prev = NULL;
        for (Edge_ID const * e = &*ew; e != EW; ++e)
        {
          if (prev != e->id)
          {
            prev = e->id;
            G.add_edge(e->a, e->b, *prev);
          }
        }
      }

      SendProgress.RestoreAndSend100Percent();
      CMT_Py_DECREF(D_npy);

      rslt_py = PyTuple_New(3);
      PyTuple_SetItem(rslt_py, 0, reinterpret_cast<PyObject *>(G.rowstart_npy));
      PyTuple_SetItem(rslt_py, 1, reinterpret_cast<PyObject *>(G.targets_npy));
      PyTuple_SetItem(rslt_py, 2, reinterpret_cast<PyObject *>(G.weights_npy));
      return rslt_py;
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(D_npy);
      CMT_Py_XDECREF(rslt_py);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(D_npy);
      CMT_Py_XDECREF(rslt_py);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(D_npy);
      CMT_Py_XDECREF(rslt_py);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(D_npy);
      CMT_Py_XDECREF(rslt_py);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }

  static PyObject *
  ncomp(PyObject * const, // self, not needed
      PyObject * const args)
  {

    PyObject * rowstart_py, *targets_py, *dummy_py;
    PyArrayObject * rowstart_npy = NULL;
    PyArrayObject * targets_npy = NULL;

    PyThreadState * PythonThreadSave = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTuple(args, "OO|O", &rowstart_py, &targets_py, &dummy_py))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (!(rowstart_npy =
          reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(rowstart_py,
          NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      if (!(targets_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          targets_py, NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }

      npy_intp const NpO = CMT_PyArray_SIZE(rowstart_npy);
      if (NpO < 1)
        throw(err_rowstartzero);
      EdgeIndex const N = NpO - 1;
      EdgeIndex const num_edges = CMT_PyArray_SIZE(targets_npy);

      Vertex
      const * const rowstart = reinterpret_cast<Vertex const *>(PyArray_DATA(
          rowstart_npy));

      EdgeIndex
      const * const targets = reinterpret_cast<EdgeIndex const *>(PyArray_DATA(
          targets_npy));

      PythonThreadSave = PyEval_SaveThread();

      Graph const graph(&targets[0], &rowstart[0], num_edges, N);
      Vertex const ncomp = boost::connected_components(graph,
          csr_graph::csr_dummy_map<Graph>());

      PyEval_RestoreThread(PythonThreadSave);
      CMT_Py_DECREF(rowstart_npy);
      CMT_Py_DECREF(targets_npy);
      return PyLong_FromSsize_t(ncomp);
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }

  static PyObject *
  Laplacian(PyObject * const, // self, not needed
      PyObject * const args)
  {
    PyObject * rowstart_py, *targets_py, *weights_py, *weighted_edges_py,
        *normalized_py;
    double eps = 0;
    double sigma_eps = 1;
    PyArrayObject * rowstart_npy = NULL;
    PyArrayObject * targets_npy = NULL;
    PyArrayObject * weights_npy = NULL;

    PyThreadState * PythonThreadSave = NULL;

    PyArrayObject * newweights_npy = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTuple(args, "OOOOddO", &rowstart_py, &targets_py,
          &weights_py, &weighted_edges_py, &eps, &sigma_eps, &normalized_py))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (!(rowstart_npy =
          reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(rowstart_py,
          NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      if (!(targets_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          targets_py, NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      if (!(weights_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          weights_py, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }

      int const weighted_edges = PyObject_IsTrue(weighted_edges_py);
      if (weighted_edges == -1)
        return NULL;

      if (eps < 0)
        throw(err_eps);
      if (sigma_eps <= 0)
        throw(err_sigma_eps);

      int const normalized = PyObject_IsTrue(normalized_py);
      if (normalized == -1)
        return NULL;

      npy_intp const NpO = CMT_PyArray_SIZE(rowstart_npy);
      if (NpO < 1)
        throw(err_rowstartzero);
      EdgeIndex const N = NpO - 1;
      EdgeIndex num_edges = CMT_PyArray_SIZE(targets_npy);
      if (num_edges != CMT_PyArray_SIZE(weights_npy))
        throw(err_samesize);

      Vertex
      const * const rowstart = reinterpret_cast<Vertex const *>(PyArray_DATA(
          rowstart_npy));

      EdgeIndex
      const * const targets = reinterpret_cast<EdgeIndex const *>(PyArray_DATA(
          targets_npy));

      Weight
      const * const weights = reinterpret_cast<Weight const *>(PyArray_DATA(
          weights_npy));

      if (!(newweights_npy =
          reinterpret_cast<PyArrayObject *>(CMT_PyArray_ZEROS(1, &num_edges,
              NPY_DOUBLE, false))))
        return NULL;
      Weight * newweights = reinterpret_cast<Weight *>(PyArray_DATA(
          newweights_npy));

      PythonThreadSave = PyEval_SaveThread();

      for (Vertex ii = 0; ii < N; ++ii)
      {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        if (targets[rowstart[ii]] != ii || weights[rowstart[ii]] != 0)
          throw(err_noloopedge);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
      }

      auto_array_ptr<Weight> degrees(N);

      if (weighted_edges)
      {
        Weight factor = -.5 / (sigma_eps * sigma_eps * eps * eps);

        Vertex r = -1;
        for (EdgeIndex ii = 0; ii < num_edges; ++ii)
        {
          // ignore diagonal edges
          if (ii == rowstart[r + 1])
          {
            ++r;
          }
          else
          {
            Weight w = weights[ii];
            w = exp(factor * w * w);
            newweights[ii] = -w;
            degrees[r] += w;
          }
        }
      }
      else
      {
        for (Vertex ii = 0; ii < N; ++ii)
        {
          degrees[ii] =
              static_cast<Weight>(rowstart[ii + 1] - rowstart[ii] - 1);
          // -1: don't count diagonal edges
        }
        std::fill(newweights, newweights + num_edges, -1);
      }

      if (normalized)
      {
        for (Weight * d = degrees; d < degrees + N; ++d)
        {
          *d = 1 / sqrt(*d);
        }

        Vertex r = -1;
        for (EdgeIndex ii = 0; ii < num_edges; ++ii)
        {
          // ignore diagonal edges
          if (ii == rowstart[r + 1])
          {
            ++r;
            newweights[ii] = 1;
          }
          else
          {
            Vertex const c = targets[ii];
            newweights[ii] *= degrees[r] * degrees[c];
          }
        }
      }
      else
      {
        for (Vertex ii = 0; ii < N; ++ii)
        {
          newweights[rowstart[ii]] = degrees[ii];
        }
      }

      PyEval_RestoreThread(PythonThreadSave);
      CMT_Py_DECREF(rowstart_npy);
      CMT_Py_DECREF(targets_npy);
      CMT_Py_DECREF(weights_npy);
      return reinterpret_cast<PyObject *>(newweights_npy);
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(newweights_npy);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(newweights_npy);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(newweights_npy);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(newweights_npy);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }

// processvertex is called as separate threads for parallel processing

  /* We have more than 9 arguments, which boost::bind cannot handle. Thus,
   we combine many arguments into a single structure.
   */
  struct processvertex_args
  {
    boost::exception_ptr * const error;
    Vertex * const vptr;
    boost::mutex * const mutex_v;
    Vertex const * const edgelist;
    Weight const * const weightlist;
    EdgeIndex const * const rowstart;
    EdgeIndex const num_edges;
    Vertex const N;
    Weight * const DD_;
  };

  static void
  processvertex(processvertex_args const A, Vertex ii)
  {
    boost::exception_ptr & error = A.error[ii];
    Vertex const N = A.N;

    Graph const graph(A.edgelist, A.rowstart, A.num_edges, N);
    auto_array_ptr<Weight> Dd(N);
    csr_graph::csr_weight_map<Graph> weight_map(A.weightlist);
    csr_graph::csr_distance_map<Graph> dist_map(&Dd[0]);

    /* To be studied in the far future (low priority): Can Dijkstra be
     speeded up if the distance map is initialized with the shortest
     paths which are already known (ie. x->y for indices x>y)?
     */
    try
    {
      while (ii < N)
      {
        //boost::dijkstra_shortest_paths
        boost::dijkstra_shortest_paths_no_color_map(graph, ii,
            boost::weight_map(weight_map).distance_map(dist_map).distance_inf(
                std::numeric_limits<Weight>::infinity()).distance_combine(
                std::plus<Weight>()).distance_zero(Weight(0))
            //.predecessor_map(csr_graph::csr_dummy_map<Graph>())
                );
        std::copy(Dd + ii + 1, Dd + N, A.DD_ + (ii * (2 * N - ii - 1) >> 1));
        A.mutex_v->lock();
        ii = *A.vptr;
        ++(*A.vptr);
        A.mutex_v->unlock();
      }
      error = boost::exception_ptr(); // no error
    }
    catch (std::exception & e)
    {
      error = boost::current_exception();
    }
  }

  static PyObject *
  graph_distance(PyObject * const, // self, not needed
      PyObject * const args,
      PyObject * const kwargs)
  {
    PyObject * rowstart_py, *targets_py, *weights_py;
    PyArrayObject * rowstart_npy = NULL;
    PyArrayObject * targets_npy = NULL;
    PyArrayObject * weights_npy = NULL;

    PyThreadState * PythonThreadSave = NULL;
    PyObject * callback = NULL;

    static char * kwlist[] =
      { arg_t1, arg_t2, arg_t3, arg_s4, NULL }; // "callback",

    PyArrayObject * Dcomp = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|O", kwlist,
          &rowstart_py, &targets_py, &weights_py, &callback))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (callback == Py_None)
      {
        callback = NULL;
      }
      else if (callback && !PyCallable_Check(callback))
      {
        throw(err_callback_not_callable);
      }
      Callback SendProgress(&PythonThreadSave, callback);

      if (!(rowstart_npy =
          reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(rowstart_py,
          NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      if (!(targets_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          targets_py, NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      if (!(weights_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          weights_py, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }

      npy_intp const NpO = CMT_PyArray_SIZE(rowstart_npy);
      if (NpO < 1)
        throw(err_rowstartzero);
      EdgeIndex const N = NpO - 1;
      EdgeIndex const num_edges = CMT_PyArray_SIZE(targets_npy);
      if (num_edges != CMT_PyArray_SIZE(weights_npy))
        throw(err_samesize);

      Vertex
      const * const rowstart = reinterpret_cast<Vertex const *>(PyArray_DATA(
          rowstart_npy));

      EdgeIndex
      const * const targets = reinterpret_cast<EdgeIndex const *>(PyArray_DATA(
          targets_npy));

      Weight
      const * const weights = reinterpret_cast<Weight const *>(PyArray_DATA(
          weights_npy));

      // Write the results from a quadratic array into a compressed array
      // (like "squareform")
      npy_intp NN = N * (static_cast<npy_intp>(N) - 1) >> 1;

      if (!(Dcomp = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(1,
          &NN, NPY_DOUBLE))))
      {
        return NULL;
      }
      Weight * DD = reinterpret_cast<Weight *>(PyArray_DATA(Dcomp));

      PythonThreadSave = PyEval_SaveThread();

      // active arrays: targets, weights, rowstart

      unsigned int const nthreads =
          multithreading ? boost::thread::hardware_concurrency() : 1;

      Graph const graph(&targets[0], &rowstart[0], num_edges, N);

      if (nthreads == 1)
      {
        auto_array_ptr<Weight> Dd(N);
        csr_graph::csr_weight_map<Graph> weight_map(&weights[0]);
        csr_graph::csr_distance_map<Graph> dist_map(&Dd[0]);

        for (Vertex ii = 0; ii < N; ++ii)
        {
          //boost::dijkstra_shortest_paths
          boost::dijkstra_shortest_paths_no_color_map(graph, ii,
              boost::weight_map(weight_map).distance_map(dist_map).distance_inf(
                  std::numeric_limits<Weight>::infinity()).distance_combine(
                  std::plus<Weight>()).distance_zero(Weight(0))
              //.predecessor_map(csr_graph::csr_dummy_map<Graph>())
                  );
          std::copy(Dd + ii + 1, Dd + N, DD);
          SendProgress(ii * 100 / N);
        }
      }
      else
      {
        boost::mutex mutex_v;
        std::vector<boost::thread *> threads(nthreads);
        Vertex v = static_cast<Vertex>(nthreads);
        std::vector<boost::exception_ptr> error(nthreads);
        processvertex_args const A =
          { &error[0], &v, &mutex_v, &targets[0], &weights[0], &rowstart[0],
              num_edges, N, DD };
        for (unsigned int ii = 0; ii < nthreads; ++ii)
        {
          threads[ii] = new boost::thread(processvertex, A, ii);
        }
        while (
#if BOOST_VERSION < 105000
        !threads[0]->timed_join(boost::posix_time::milliseconds(200))
#else
        !threads[0]->try_join_for(boost::chrono::milliseconds(200))
#endif
        )
        {
          SendProgress(v * 100 / N);
        }
        for (unsigned int ii = 0; ii < nthreads; ++ii)
        {
          if (ii > 0)
            threads[ii]->join();
          if (error[ii])
          {
            boost::rethrow_exception(error[ii]);
          }
          delete threads[ii];
        }
      }

      SendProgress.RestoreAndSend100Percent();
      CMT_Py_DECREF(rowstart_npy);
      CMT_Py_DECREF(targets_npy);
      CMT_Py_DECREF(weights_npy);
      return reinterpret_cast<PyObject *>(Dcomp);
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(Dcomp);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(Dcomp);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(Dcomp);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rowstart_npy);
      CMT_Py_XDECREF(targets_npy);
      CMT_Py_XDECREF(weights_npy);
      CMT_Py_XDECREF(Dcomp);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

namespace conn_comp
{

  typedef npy_intp t_index;

  class union_find
  {
  private:
    auto_array_ptr<t_index> const parent;
    auto_array_ptr<t_index> const sizes;

    // Noncopyable
    union_find(union_find const &);
    union_find &
    operator=(union_find const &);

  public:
    t_index ncomp;

    union_find(const t_index size)
        : parent(size, -1),
          sizes(size, -1),
          ncomp(size)
    {
    }

    ~union_find()
    {
    }

    t_index
    Find(t_index idx) const
    {
      if (parent[idx] != -1)
      { // a → b
        t_index p = idx;
        idx = parent[idx];
        if (parent[idx] != -1)
        { // a → b → c
          do
          {
            idx = parent[idx];
          }
          while (parent[idx] != -1);
          do
          {
            t_index tmp = parent[p];
            parent[p] = idx;
            p = tmp;
          }
          while (parent[p] != idx);
        }
      }
      return idx;
    }

    void
    Union(t_index node1, t_index node2)
    {
      node1 = Find(node1);
      node2 = Find(node2);
      if (node1 == node2)
        return;
      if (sizes[node1] < sizes[node2])
      {
        parent[node1] = node2;
        sizes[node2] += sizes[node1];
      }
      else
      {
        parent[node2] = node1;
        sizes[node1] += sizes[node2];
      }
      --ncomp;
    }
  };

  static PyObject *
  _conn_comp_loop(PyObject * const, // self, not needed
      PyObject * j_py)
  {

    PyArrayObject * j_npy = NULL;

    PyThreadState * PythonThreadSave = NULL;

    PyObject * rslt_py = NULL;

    try
    {
      if (!(j_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(j_py,
          NPY_INT, 2, 2, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }

      npy_int * j = reinterpret_cast<npy_int *>(PyArray_DATA(j_npy));

      if (PyArray_DIM(j_npy, 0) > NPY_MAX_INT
          || PyArray_DIM(j_npy, 1) > NPY_MAX_INT)
      {
        throw(err_toomanypoints);
      }
      t_index const N = PyArray_DIM(j_npy, 0);
      t_index const k = PyArray_DIM(j_npy, 1);

      PythonThreadSave = PyEval_SaveThread();

      union_find UF(N);
      t_index kk;
      for (kk = 1; kk < k; ++kk)
      {
        for (t_index jj = 0; jj < N; ++jj)
        {
          UF.Union(jj, j[jj * k + kk]);
          /* Is it faster with or without this line - I did not test it.
           */
          //if (UF.ncomp==1) break;
        }
        if (UF.ncomp == 1)
          break;
      }

      PyEval_RestoreThread(PythonThreadSave);
      CMT_Py_DECREF(j_npy);

      rslt_py = PyTuple_New(2);
      PyTuple_SetItem(rslt_py, 0, PyLong_FromSsize_t(UF.ncomp));
      PyTuple_SetItem(rslt_py, 1, PyLong_FromSsize_t(kk));
      return rslt_py;
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(j_npy);
      CMT_Py_XDECREF(rslt_py);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(j_npy);
      CMT_Py_XDECREF(rslt_py);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(j_npy);
      CMT_Py_XDECREF(rslt_py);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(j_npy);
      CMT_Py_XDECREF(rslt_py);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

namespace nn_from_dm
{

  typedef npy_intp t_index;

// Indexing function
// D is the upper triangular part of a symmetric (NxN)-matrix
// We require r_ < c_ !
#define X_(r_,c_) ( X[(static_cast<std::ptrdiff_t>(2*N-3-(r_))*(r_)>>1)\
                        +(c_)-1] )

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#endif
  class dist_idx
  { // will be copied...
  public:
    npy_double d;
    npy_int j;

    bool
    operator<(const dist_idx& b) const
    {
      return d < b.d;
    }
  };
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// processrow is called as separate threads for parallel processing

  static void
  processrow(
      boost::exception_ptr & error,
      npy_int ii,
      npy_int * const iptr,
      boost::mutex * mutex_i,
      npy_double const * const X,
      npy_intp const N,
      npy_intp const k,
      npy_double * const d,
      npy_int * const j)
  {
    try
    {
      auto_array_ptr<dist_idx> DJ(N);
      while (ii < N)
      {
        for (npy_int jj = 0; jj < ii; ++jj)
        {
          DJ[jj].d = X_(jj,ii);
          DJ[jj].j = jj;
        }
        DJ[ii].d = 0;
        DJ[ii].j = ii;
        for (npy_int jj = ii + 1; jj < N; ++jj)
        {
          DJ[jj].d = X_(ii,jj);
          DJ[jj].j = jj;
        }
        std::partial_sort(&DJ[0], &DJ[k], &DJ[N]);
        for (npy_int jj = 0; jj < k; ++jj)
        {
          d[ii * k + jj] = DJ[jj].d;
          j[ii * k + jj] = DJ[jj].j;
        }

        mutex_i->lock();
        ii = *iptr;
        ++(*iptr);
        mutex_i->unlock();
      }
      error = boost::exception_ptr(); // no error
    }
    catch (std::exception & e)
    {
      error = boost::current_exception();
    }
  }

  static PyObject *
  nearest_neighbors_from_dm(PyObject * const, // self, not needed
      PyObject * const args,
      PyObject * const kwargs)
  {

    PyObject * X_py;
    PyArrayObject * X_npy = NULL;

    Py_ssize_t k;

    PyThreadState * PythonThreadSave = NULL;
    PyObject * callback = NULL;

    static char * kwlist[] =
      { arg_t4, arg_t5, arg_s4, NULL };

    PyObject * rslt = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "On|O", kwlist, &X_py, &k,
          &callback))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (!(X_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(X_py,
          NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }

      npy_double * X = reinterpret_cast<npy_double *>(PyArray_DATA(X_npy));

      npy_intp const N = n_obs(PyArray_DIM(X_npy, 0));

      if (k < 1 || k > N)
        throw(err_k);

      if (callback == Py_None)
      {
        callback = NULL;
      }
      else if (callback && !PyCallable_Check(callback))
      {
        throw(err_callback_not_callable);
      }
      Callback SendProgress(&PythonThreadSave, callback);

      npy_intp dims[2] =
        { N, k };
      PyArrayObject * d_npy;
      if (!(d_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(2,
          dims, NPY_DOUBLE))))
      {
        return NULL;
      }
      npy_double * const d = reinterpret_cast<npy_double *>(PyArray_DATA(d_npy));
      PyArrayObject * j_npy;
      if (!(j_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(2,
          dims, NPY_INT))))
      {
        return NULL;
      }
      npy_int * const j = reinterpret_cast<npy_int *>(PyArray_DATA(j_npy));

      PythonThreadSave = PyEval_SaveThread();

      unsigned int const nthreads =
          multithreading ? boost::thread::hardware_concurrency() : 1;

      if (nthreads == 1)
      {
        auto_array_ptr<dist_idx> DJ(N);

        for (npy_int ii = 0; ii < N; ++ii)
        {
          for (npy_int jj = 0; jj < ii; ++jj)
          {
            DJ[jj].d = X_(jj,ii);
            DJ[jj].j = jj;
          }
          DJ[ii].d = 0;
          DJ[ii].j = ii;
          for (npy_int jj = ii + 1; jj < N; ++jj)
          {
            DJ[jj].d = X_(ii,jj);
            DJ[jj].j = jj;
          }
          std::partial_sort(&DJ[0], &DJ[k], &DJ[N]);
          for (npy_int jj = 0; jj < k; ++jj)
          {
            d[ii * k + jj] = DJ[jj].d;
            j[ii * k + jj] = DJ[jj].j;
          }
          SendProgress(ii * 100 / N);
        }

      }
      else
      {
        boost::mutex mutex_i;
        std::vector<boost::thread *> threads(nthreads);
        npy_int row = static_cast<npy_int>(nthreads);
        std::vector<boost::exception_ptr> error(nthreads);
        for (unsigned int ii = 0; ii < nthreads; ++ii)
        {
          threads[ii] = new boost::thread(processrow, boost::ref(error[ii]), ii,
              &row, &mutex_i, X, N, k, d, j);
        }
        while (
#if BOOST_VERSION < 105000
        !threads[0]->timed_join(boost::posix_time::milliseconds(200))
#else
        !threads[0]->try_join_for(boost::chrono::milliseconds(200))
#endif
        )
        {
          SendProgress(row * 100 / N);
        }
        for (unsigned int i = 0; i < nthreads; ++i)
        {
          if (i > 0)
            threads[i]->join();
          if (error[i])
          {
            boost::rethrow_exception(error[i]);
          }
          delete threads[i];
        }
      }

      SendProgress(100);

      PyEval_RestoreThread(PythonThreadSave);
      CMT_Py_DECREF(X_npy);

      rslt = PyTuple_New(2);
      PyTuple_SetItem(rslt, 0, reinterpret_cast<PyObject *>(d_npy));
      PyTuple_SetItem(rslt, 1, reinterpret_cast<PyObject *>(j_npy));
      return rslt;
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(X_npy);
      CMT_Py_XDECREF(rslt);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(X_npy);
      CMT_Py_XDECREF(rslt);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(X_npy);
      CMT_Py_XDECREF(rslt);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(rslt);
      CMT_Py_XDECREF(X_npy);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

namespace compressed_submatrix
{

  static PyObject *
  compressed_submatrix(PyObject * const, // self, not needed
      PyObject * const args)
  {

    PyObject * dm_py;
    PyArrayObject * dm_npy = NULL;
    PyObject * idx_py;
    PyArrayObject * idx_npy = NULL;

    PyThreadState * PythonThreadSave = NULL;
    PyObject * callback = NULL;

    PyArrayObject * rslt_npy = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTuple(args, "OO", &dm_py, &idx_py, &callback))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      if (!(dm_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          dm_py, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }
      if (!(idx_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
          idx_py,
          NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))))
      {
        return NULL;
      }

      npy_double * dm = reinterpret_cast<npy_double *>(PyArray_DATA(dm_npy));
      npy_intp * idx = reinterpret_cast<npy_intp *>(PyArray_DATA(idx_npy));

      npy_intp const N = n_obs(PyArray_DIM(dm_npy, 0));
      npy_intp const n = PyArray_DIM(idx_npy, 0);

      if (n > N)
        throw(err_n);

      npy_intp nn = n * (n - 1) >> 1;

      if (!(rslt_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(
          1, &nn, NPY_DOUBLE))))
      {
        return NULL;
      }
      npy_double * const rslt = reinterpret_cast<npy_double *>(PyArray_DATA(
          rslt_npy));

      PythonThreadSave = PyEval_SaveThread();

      npy_intp jj = 0;
      for (npy_intp r = 0; r < n - 1; ++r)
      {
        npy_intp const u = (2 * N - 3 - idx[r]) * idx[r] / 2 - 1;
        for (npy_intp c = r + 1; c < n; ++c)
        {
          rslt[jj] = dm[u + idx[c]];
          ++jj;
        }
      }

      PyEval_RestoreThread(PythonThreadSave);
      CMT_Py_DECREF(dm_npy);
      CMT_Py_DECREF(idx_npy);
      return reinterpret_cast<PyObject *>(rslt_npy);
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(dm_npy);
      CMT_Py_XDECREF(idx_npy);
      CMT_Py_XDECREF(rslt_npy);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(dm_npy);
      CMT_Py_XDECREF(idx_npy);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(dm_npy);
      CMT_Py_XDECREF(idx_npy);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(dm_npy);
      CMT_Py_XDECREF(idx_npy);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

namespace fcluster
{

  static PyObject *
  fcluster(PyObject * const, // self, not needed
      PyObject * const args)
  {

    PyObject * Z_py;
    PyArrayObject * Z_npy = NULL;
    Py_ssize_t num_clust;

    PyThreadState * PythonThreadSave = NULL;
    PyObject * callback = NULL;

    PyArrayObject * rslt_npy = NULL;

    try
    {
      // Parse the input arguments
      if (!PyArg_ParseTuple(args, "On", &Z_py, &num_clust, &callback))
      {
        return NULL; // Error if the arguments have the wrong type.
      }

      Py_ssize_t N = PyObject_Length(Z_py) + 1;
      if (N == 0)
        return NULL;

      if (!(rslt_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_SimpleNew(
          1, &N, NPY_INT))))
      {
        return NULL;
      }
      npy_int * const rslt = reinterpret_cast<npy_int *>(PyArray_DATA(rslt_npy));

      if (num_clust == 1)
      {
        std::fill_n(rslt, N, 0);
      }
      else
      {
        if (!(Z_npy = reinterpret_cast<PyArrayObject *>(CMT_PyArray_FROMANY(
            Z_py, NPY_DOUBLE, 2, 2, NPY_ARRAY_ALIGNED))))
        {
          return NULL;
        }

        PythonThreadSave = PyEval_SaveThread();

        if (num_clust < 1 || num_clust > N)
          throw(err_num_clust);

        auto_array_ptr<npy_intp> parent(2 * N - num_clust, -1);
        npy_intp end_N = N - num_clust;

        for (npy_intp ii = 0; ii < end_N; ++ii)
        {
          npy_intp const a =
              static_cast<npy_intp const>(*reinterpret_cast<npy_double const *>(CMT_PyArray_GETPTR2(
                  Z_npy, ii, 0)));
          npy_intp const b =
              static_cast<npy_intp const>(*reinterpret_cast<npy_double const *>(CMT_PyArray_GETPTR2(
                  Z_npy, ii, 1)));
          parent[a] = parent[b] = N + ii;
        }

        auto_array_ptr<npy_intp> clust(N);

        for (npy_intp ii = 0; ii < N; ++ii)
        {
          npy_intp idx = ii;
          if (parent[idx] != -1)
          {
            npy_intp p = idx;
            idx = parent[idx];
            if (parent[idx] != -1)
            {
              do
              {
                idx = parent[idx];
              }
              while (parent[idx] != -1);
              do
              {
                npy_intp tmp = p;
                p = parent[p];
                parent[tmp] = idx;
              }
              while (parent[p] != idx);
            }
          }
          clust[ii] = idx;
        }

        // clust contains cluster assignments, but not necessarily numbered
        // 0...num_clust-1. Relabel the clusters.

        auto_array_ptr<npy_intp> idx(N);
        std::copy(clust + 0, clust + N, idx + 0);
        std::sort(idx + 0, idx + N);
        npy_intp const * const endidx = std::unique(idx + 0, idx + N);
        auto_array_ptr<npy_int> idx2(2 * N - num_clust);
        {
          npy_int ii = 0;
          for (npy_intp const * idxptr = idx + 0; idxptr < endidx;
              ++idxptr, ++ii)
          {
            idx2[*idxptr] = ii;
          }
        }
        for (npy_intp ii = 0; ii < N; ++ii)
        {
          rslt[ii] = idx2[clust[ii]];
        }

        PyEval_RestoreThread(PythonThreadSave);
        CMT_Py_DECREF(Z_npy);
      }

      return reinterpret_cast<PyObject *>(rslt_npy);
    }
    catch (std::bad_alloc&)
    {
      if (PythonThreadSave)
      { // Only restore if the state has been saved
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(Z_npy);
      CMT_Py_XDECREF(rslt_npy);
      return PyErr_NoMemory();
    }
    catch (std::exception& e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(Z_npy);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(PyExc_EnvironmentError, e.what());
      return NULL;
    }
    catch (errormessage e)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(Z_npy);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(e.errortype, e.message);
      return NULL;
    }
    catch (...)
    {
      if (PythonThreadSave)
      {
        PyEval_RestoreThread(PythonThreadSave);
      }
      CMT_Py_XDECREF(Z_npy);
      CMT_Py_XDECREF(rslt_npy);
      PyErr_SetString(PyExc_EnvironmentError,
          "C++ exception (unknown reason). Please send a bug report.");
      return NULL;
    }
  }
}

/*
 Python interface code
 */

// List the C++ methods that this extension provides.
static PyMethodDef cmappertoolsMethods[] =
      {
        { "eccentricity",
            reinterpret_cast<PyCFunction>(kernel_filter::eccentricity),
            METH_VARARGS | METH_KEYWORDS, NULL },
        { "Gauss_density",
            reinterpret_cast<PyCFunction>(kernel_filter::Gauss_density),
            METH_VARARGS | METH_KEYWORDS, NULL },
        { "neighborhood_graph",
            reinterpret_cast<PyCFunction>(graph::neighborhood_graph),
            METH_VARARGS | METH_KEYWORDS, NULL },
        { "Laplacian", reinterpret_cast<PyCFunction>(graph::Laplacian),
        METH_VARARGS, NULL },
        { "ncomp", reinterpret_cast<PyCFunction>(graph::ncomp), METH_VARARGS,
        NULL },
        { "graph_distance",
            reinterpret_cast<PyCFunction>(graph::graph_distance),
            METH_VARARGS | METH_KEYWORDS, NULL },
        { "_conn_comp_loop",
            reinterpret_cast<PyCFunction>(conn_comp::_conn_comp_loop), METH_O,
            NULL },
            { "nearest_neighbors_from_dm",
                reinterpret_cast<PyCFunction>(nn_from_dm::nearest_neighbors_from_dm),
                METH_VARARGS | METH_KEYWORDS, NULL },
            { "compressed_submatrix",
                reinterpret_cast<PyCFunction>(compressed_submatrix::compressed_submatrix),
                METH_VARARGS, NULL },
            { "fcluster", reinterpret_cast<PyCFunction>(fcluster::fcluster),
            METH_VARARGS, NULL },
            { NULL, NULL, 0, NULL } // Sentinel - marks the end of this structure
      };

/* Tell Python about these methods.

 Python 2.x and 3.x differ in their C APIs for this part.
 */
#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef cmappertoolsmodule =
  {
  PyModuleDef_HEAD_INIT, "cmappertools",
NULL, // no module documentation
      -1, /* size of per-interpreter state of the module,
       or -1 if the module keeps state in global variables. */
      cmappertoolsMethods };

#if HAVE_VISIBILITY
#pragma GCC visibility push(default)
#endif

PyMODINIT_FUNC PyInit_cmappertools(void)
{
  PyObject * const m = PyModule_Create(&cmappertoolsmodule);
  if (!m)
  {
    return NULL;
  }
  if (PyModule_AddStringConstant(m, "__version__", __version__))
  {
    return NULL;
  }
// Must be present for NumPy. Called first after above line.
  import_array();
  return m;
}

#if HAVE_VISIBILITY
#pragma GCC visibility pop
#endif

# else // Python 2.x

#if HAVE_VISIBILITY
#pragma GCC visibility push(default)
#endif

PyMODINIT_FUNC
initcmappertools(void)
{
  PyObject * const m = Py_InitModule("cmappertools", cmappertoolsMethods);
  if (m)
  {
    if (PyModule_AddStringConstant(m, "__version__", __version__))
    {
      return;
    }

    // Must be present for NumPy. Called first after above line.
    import_array();
  }
}

#if HAVE_VISIBILITY
#pragma GCC visibility pop
#endif

#endif // PY_VERSION

#if HAVE_VISIBILITY
#pragma GCC visibility pop
#endif
