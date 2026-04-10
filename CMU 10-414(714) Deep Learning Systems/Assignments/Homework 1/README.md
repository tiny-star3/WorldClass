# Homework 1

**作业原址**：[dlsyscourse/hw1](https://github.com/dlsyscourse/hw1)
非常感谢老师的付出和开源，以下是我的实现(特别感谢 Google AI Studio 提供远程指导😝)   

## Question 1: Implementing forward computation

```python
class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION
        

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.resize(a, self.shape)
        ### END YOUR SOLUTION


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axis1, axis2 = self.axes if self.axes else (a.ndim - 2, a.ndim - 1)
        return array_api.swapaxes(a, axis1, axis2)
        ### END YOUR SOLUTION


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "forward"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw1
plugins: typeguard-4.5.1, anyio-4.12.1, langsmith-0.7.18
collected 29 items / 17 deselected / 12 selected                               

tests/hw1/test_autograd_hw.py::test_power_scalar_forward PASSED          [  8%]
tests/hw1/test_autograd_hw.py::test_divide_forward PASSED                [ 16%]
tests/hw1/test_autograd_hw.py::test_divide_scalar_forward PASSED         [ 25%]
tests/hw1/test_autograd_hw.py::test_matmul_forward PASSED                [ 33%]
tests/hw1/test_autograd_hw.py::test_summation_forward PASSED             [ 41%]
tests/hw1/test_autograd_hw.py::test_broadcast_to_forward PASSED          [ 50%]
tests/hw1/test_autograd_hw.py::test_reshape_forward PASSED               [ 58%]
tests/hw1/test_autograd_hw.py::test_negate_forward PASSED                [ 66%]
tests/hw1/test_autograd_hw.py::test_transpose_forward PASSED             [ 75%]
tests/hw1/test_autograd_hw.py::test_log_forward PASSED                   [ 83%]
tests/hw1/test_autograd_hw.py::test_exp_forward PASSED                   [ 91%]
tests/hw1/test_autograd_hw.py::test_ewisepow_forward PASSED              [100%]

====================== 12 passed, 17 deselected in 0.54s =======================
```

## Question 2: Implementing backward computation

```python
class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad*rhs*power(lhs, add_scalar(rhs, -1)), out_grad*node*log(lhs)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*mul_scalar(power_scalar(node.inputs[0], self.scalar-1), self.scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return divide(out_grad, rhs), out_grad*(negate(lhs)*power_scalar(rhs, -2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axis1, axis2 = self.axes if self.axes else (a.ndim - 2, a.ndim - 1)
        return array_api.swapaxes(a, axis1, axis2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 求导逻辑：既然你把第 (i,j) 个元素搬到了 (j,i)，
        # 那么反向传播时，第 (j,i) 个位置的梯度就要搬回 (i,j)
        # 转置的导数就是梯度的转置
        if self.axes:
          return transpose(out_grad, self.axes)
        else:
          return transpose(out_grad)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 求导逻辑：不管你排版怎么变，梯度就是“影响程度”
        # 既然数值没变，那反向传回来的梯度排版必须变回原来的样子，才能和原来的矩阵 X 形状对齐。
        # 变形的导数就是把梯度变形回初始形状
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 广播的导数，就是把对应维度上的梯度全部加起来
        axes = []
        n_in = len(node.inputs[0].shape)
        n_out = len(self.shape)
        for i in range(n_out):
          if i>=n_in or node.inputs[0].shape[i]==1:
            axes.append(i)
        res = summation(out_grad, tuple(axes))
        return reshape(res, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Needle 的 broadcast_to 要求维度的数量必须匹配
        # 在 Summation 中，如果 axis 被求和了，那个维度就消失了
        # 构造一个把被 sum 掉的轴变成 1 的形状
        new_shape = list(node.inputs[0].shape)
        if self.axes is not None:
          # 兼容 int 和 tuple
          axes = [self.axes] if isinstance(self.axes, int) else self.axes
          for ax in axes:
            new_shape[ax] = 1
        else:
          new_shape = [1] * len(node.inputs[0].shape)
        return broadcast_to(reshape(out_grad, new_shape), node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Batched MatMul：指带有“批次（Batch）”维度的乘法，比如 (B,M,N)×(N,K)=(B,M,K)
        # 或者更复杂的 (B1,B2,M,N)×(B1,B2,N,K)=(B1,B2,M,K)
        # 广播机制（Broadcasting）的存在，导致反向传播时需要对多出来的维度进行“求和（Summation）”
        lhs, rhs = node.inputs
        resl = matmul(out_grad, transpose(rhs))
        resr = matmul(transpose(lhs), out_grad)
        if len(lhs.shape)<len(out_grad.shape):
          resl = summation(resl, tuple(range(len(out_grad.shape)-len(lhs.shape)))).reshape(lhs.shape)
        if len(rhs.shape)<len(out_grad.shape):
          resr = summation(resr, tuple(range(len(out_grad.shape)-len(rhs.shape)))).reshape(rhs.shape)
        return resl, resr
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*node.cached_data
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


```

Reshape (变形)：只是重新排列，总元素个数必须相等
BroadcastTo (广播)：发生了（虚拟）复制，总元素个数通常会变多

```bash
!python3 -m pytest -l -v -k "backward"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw1
plugins: typeguard-4.5.1, anyio-4.12.1, langsmith-0.7.18
collected 29 items / 16 deselected / 13 selected                               

tests/hw1/test_autograd_hw.py::test_power_scalar_backward PASSED         [  7%]
tests/hw1/test_autograd_hw.py::test_divide_backward PASSED               [ 15%]
tests/hw1/test_autograd_hw.py::test_divide_scalar_backward PASSED        [ 23%]
tests/hw1/test_autograd_hw.py::test_matmul_simple_backward PASSED        [ 30%]
tests/hw1/test_autograd_hw.py::test_matmul_batched_backward PASSED       [ 38%]
tests/hw1/test_autograd_hw.py::test_reshape_backward PASSED              [ 46%]
tests/hw1/test_autograd_hw.py::test_negate_backward PASSED               [ 53%]
tests/hw1/test_autograd_hw.py::test_transpose_backward PASSED            [ 61%]
tests/hw1/test_autograd_hw.py::test_broadcast_to_backward PASSED         [ 69%]
tests/hw1/test_autograd_hw.py::test_summation_backward PASSED            [ 76%]
tests/hw1/test_autograd_hw.py::test_log_backward PASSED                  [ 84%]
tests/hw1/test_autograd_hw.py::test_exp_backward PASSED                  [ 92%]
tests/hw1/test_autograd_hw.py::test_ewisepow_backward PASSED             [100%]

====================== 13 passed, 16 deselected in 0.61s =======================
```

## Question 3: Topological sort

```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    topo_order = []
    visited = set()
    for node in node_list:
      if node not in visited:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    visited.add(node)
    for input in node.inputs:
      if input not in visited:
        topo_sort_dfs(input, visited, topo_order)
    topo_order.append(node)
    ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -k "topo_sort"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw1
plugins: typeguard-4.5.1, anyio-4.12.1, langsmith-0.7.18
collected 29 items / 28 deselected / 1 selected                                

tests/hw1/test_autograd_hw.py .                                          [100%]

======================= 1 passed, 28 deselected in 0.60s =======================
```

## Question 4: Implementing reverse mode differentiation

```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for i in reverse_topo_order:
      # 容错处理：如果当前节点没有接收到梯度，直接跳过(比如，常量)
      if i not in node_to_output_grads_list:
        continue
      grads_list = node_to_output_grads_list[i]
      # 使用自己重载的 + (Tensor.__add__)
      vi = grads_list[0]
      for grad in grads_list[1:]:
        vi = vi + grad
      i.grad = vi
      # 判断是否是叶子节点
      if i.op is not None:
        vki = i.op.gradient_as_tuple(vi, i)
        for index, k in enumerate(i.inputs):
          if k not in node_to_output_grads_list:
            node_to_output_grads_list[k] = []
          node_to_output_grads_list[k].append(vki[index])
        
    ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -k "compute_gradient"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw1
plugins: typeguard-4.5.1, anyio-4.12.1, langsmith-0.7.18
collected 29 items / 28 deselected / 1 selected                                

tests/hw1/test_autograd_hw.py .                                          [100%]

======================= 1 passed, 28 deselected in 0.57s =======================
```

## Question 5: Softmax loss

```python
def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(label_filename, 'rb') as f:
      magicNumber, itemsNumber = struct.unpack('>ii', f.read(8))
      labels = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(image_filename, 'rb') as f:
      magicNumber, itemsNumber, rowNumber, colNumber = struct.unpack('>iiii', f.read(16))
      images = np.frombuffer(f.read(), dtype=np.uint8)
      images = images.reshape(itemsNumber, rowNumber * colNumber)
      images = images.astype(np.float32) / 255

    return (images, labels)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return ndl.summation(ndl.add(ndl.log(ndl.summation(ndl.exp(Z), axes=1)), ndl.negate(ndl.summation(ndl.multiply(Z, y_one_hot), axes=1))))/Z.shape[0]
    ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -k "softmax_loss_ndl"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw1
plugins: typeguard-4.5.1, anyio-4.12.1, langsmith-0.7.18
collected 29 items / 28 deselected / 1 selected                                

tests/hw1/test_autograd_hw.py .                                          [100%]

======================= 1 passed, 28 deselected in 1.52s =======================
```

## Question 6: SGD for a two-layer neural network

```python
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.where(a>0, a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, Tensor(node.realize_cached_data()>0, device=out_grad.device))
        ### END YOUR SOLUTION
```

```python
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    # 按照 batch 步长遍历
    for i in range(0, num_examples, batch):
      # 获取当前 batch 数据
      # 处理最后一个 batch 可能不足 batch size 的情况
      X_batch = X[i : i + batch]
      y_batch = y[i : i + batch]
      X_batch_tensor = ndl.Tensor(X_batch)
      y_one_hot = np.zeros((X_batch.shape[0], num_classes))
      y_one_hot[np.arange(X_batch.shape[0]), y_batch] = 1
      y_one_hot_tensor = ndl.Tensor(y_one_hot)
      logits = ndl.relu(X_batch_tensor @ W1)@ W2
      loss = softmax_loss(logits, y_one_hot_tensor)
      loss.backward()
      # 使用 .data 来确保更新操作不被计入计算图
      W1.data = W1.data - lr * W1.grad.data
      W2.data = W2.data - lr * W2.grad.data
    return W1, W2
    ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -l -k "nn_epoch_ndl"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/drive/MyDrive/10714/hw1
plugins: typeguard-4.5.1, anyio-4.12.1, langsmith-0.7.18
collected 29 items / 28 deselected / 1 selected                                

tests/hw1/test_autograd_hw.py .                                          [100%]

======================= 1 passed, 28 deselected in 4.73s =======================
```

