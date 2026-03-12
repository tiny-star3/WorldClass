#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace py = pybind11;

// m*n @ n*k
void matrixMul(const float *x, const float *y, float *result, size_t m, size_t n, size_t k)
{
  for(size_t i=0; i<m; i++)
  {
    for(size_t j=0; j<k; j++)
    {
      for(size_t t=0; t<n; t++)
      {
        result[i*k+j] += x[i*n+t]*y[t*k+j];
      }
    }
  }
}

void matrixT(const float *x, float *result, size_t m, size_t n)
{
  for(size_t i=0; i<m; i++)
  {
    for(size_t j=0; j<n; j++)
    {
      result[j*m+i] = x[i*n+j];
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float* Z = new float[batch*k]();
    float* XT = new float[batch*n]();
    float* loss = new float[batch*n]();
    for(size_t i=0; i<m; i+=batch)
    {
      size_t batchSize = std::min(m-i, batch);
      memset(Z, 0, batch*k*sizeof(float));
      matrixMul(X+i*n, theta, Z, batchSize, n, k);
      for(size_t j=0; j<batchSize; j++)
      {
        for(size_t t=0; t<k; t++)
        {
          Z[j*k+t] = exp(Z[j*k+t]);
        }
      }
      for(size_t j=0; j<batchSize; j++)
      {
        float sum=0;
        for(size_t t=0; t<k; t++)
        {
          sum+=Z[j*k+t];
        }
        for(size_t t=0; t<k; t++)
        {
          Z[j*k+t]/=sum;
          if(t==y[i+j]) Z[j*k+t]--;
        }
      }
      memset(XT, 0, batch*n*sizeof(float));
      matrixT(X+i*n, XT, batchSize, n);
      memset(loss, 0, batch*n*sizeof(float));
      matrixMul(XT, Z, loss, n, batchSize, k);
      for(size_t j=0; j<n; j++)
      {
        for(size_t t=0; t<k; t++)
        {
          loss[j*k+t]/=batchSize;
          theta[j*k+t]-=lr*loss[j*k+t];
        }
      }

    }
    delete[] Z;
    delete[] XT;
    delete[] loss;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
