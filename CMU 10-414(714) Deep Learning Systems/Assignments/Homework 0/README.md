# Homework 0
**作业原址**：[dlsyscourse/hw0](https://github.com/dlsyscourse/hw0)
非常感谢老师的付出和开源，以下是我的实现(特别感谢 Google AI Studio 提供远程指导😝)   

## Question 1: A basic `add` function, and testing/autograding basics

```python
def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE
```

```bash
!python3 -m pytest -k "add"
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw0
plugins: anyio-4.12.1, typeguard-4.5.1, langsmith-0.7.7
collected 6 items / 5 deselected / 1 selected                                  

tests/test_simple_ml.py .                                                [100%]

======================= 1 passed, 5 deselected in 0.44s ========================
```

## Question 2: Loading MNIST data

```python
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(label_filename, 'rb') as f:
      magicNumber, itemsNumber = struct.unpack('>ii', f.read(8))
      labels = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(image_filename, 'rb') as f:
      magicNumber, itemsNumber, rowNumber, colNumber = struct.unpack('>iiii', f.read(16))
      images = np.frombuffer(f.read(), dtype=np.uint8)
      images = images.reshape(itemsNumber, rowNumber * colNumber)
      images = images.astype(np.float32) / 255

    return (images, labels)

    ### END YOUR CODE
```

```bash
!python3 -m pytest -k "parse_mnist"
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw0
plugins: anyio-4.12.1, typeguard-4.5.1, langsmith-0.7.7
collected 6 items / 5 deselected / 1 selected                                  

tests/test_simple_ml.py .                                                [100%]

======================= 1 passed, 5 deselected in 1.42s ========================
```

## Question 3: Softmax loss

```python
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.average(np.subtract(np.log(np.sum(np.exp(Z), axis=1)), Z[np.arange(Z.shape[0]), y]))
    ### END YOUR CODE
```

```bash
!python3 -m pytest -k "softmax_loss"
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw0
plugins: anyio-4.12.1, typeguard-4.5.1, langsmith-0.7.7
collected 6 items / 5 deselected / 1 selected                                  

tests/test_simple_ml.py .                                                [100%]

======================= 1 passed, 5 deselected in 1.11s ========================
```

## Question 4: Stochastic gradient descent for softmax regression

```python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = theta.shape[1]
    # 按照 batch 步长遍历
    for i in range(0, num_examples, batch):
      # 获取当前 batch 数据
      # 处理最后一个 batch 可能不足 batch size 的情况
      X_batch = X[i : i + batch]
      y_batch = y[i : i + batch]
      curr_batch_size = X_batch.shape[0]
      Iy = np.eye(len(X_batch), len(theta[0]))[y_batch]
      Z = np.exp(X_batch @ theta)
      Z = Z / np.sum(Z, axis=1, keepdims=True)
      loss = (X_batch.T @ np.subtract(Z, Iy)) / len(X_batch)
      theta -= lr*loss
    ### END YOUR CODE
```

```bash
!python3 -m pytest -k "softmax_regression_epoch and not cpp"
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw0
plugins: anyio-4.12.1, langsmith-0.7.13, typeguard-4.5.1
collected 6 items / 5 deselected / 1 selected                                  

tests/test_simple_ml.py .                                                [100%]

======================= 1 passed, 5 deselected in 0.98s ========================
```

## Question 5: SGD for a two-layer neural network

```python
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    # 按照 batch 步长遍历
    for i in range(0, num_examples, batch):
      # 获取当前 batch 数据
      # 处理最后一个 batch 可能不足 batch size 的情况
      X_batch = X[i : i + batch]
      y_batch = y[i : i + batch]
      Z1 = np.maximum(0, X_batch @ W1)
      Iy = np.eye(len(X_batch), num_classes)[y_batch]
      G2 = np.exp(Z1 @ W2)
      G2 = G2 / np.sum(G2, axis=1, keepdims=True)
      G2 = G2 - Iy
      G1 = np.where(Z1>0, 1, 0) * (G2 @ W2.T)
      lossW1 = (X_batch.T @ G1) / len(X_batch)
      lossW2 = (Z1.T @ G2) / len(X_batch)
      W1 -= lr * lossW1
      W2 -= lr * lossW2
    ### END YOUR CODE
```

```bash
!python3 -m pytest -k "nn_epoch"
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw0
plugins: anyio-4.12.1, langsmith-0.7.13, typeguard-4.5.1
collected 6 items / 5 deselected / 1 selected                                  

tests/test_simple_ml.py .                                                [100%]

======================= 1 passed, 5 deselected in 3.00s ========================
```

## Question 6: Softmax regression in C++

```C++
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
```

```bash
!make
!python3 -m pytest -k "softmax_regression_epoch_cpp"
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw0
plugins: anyio-4.12.1, langsmith-0.7.13, typeguard-4.5.1
collected 6 items / 5 deselected / 1 selected                                  

tests/test_simple_ml.py .                                                [100%]

======================= 1 passed, 5 deselected in 0.95s ========================
```

