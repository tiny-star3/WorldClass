# Homework 2

**作业原址**：[dlsyscourse/hw2](https://github.com/dlsyscourse/hw2)
非常感谢老师的付出和开源，以下是我的实现(特别感谢 Google AI Studio 提供远程指导😝)   

## Question 1

```python
def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain*(math.sqrt(6/(fan_in+fan_out)))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain*math.sqrt(2/(fan_in+fan_out))
    return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(2)*math.sqrt(3/fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2)/math.sqrt(fan_in)
    return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_init"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: anyio-4.12.1, typeguard-4.5.1, langsmith-0.7.18
collected 93 items / 89 deselected / 4 selected                                

tests/hw2/test_nn_and_optim.py::test_init_kaiming_uniform PASSED         [ 25%]
tests/hw2/test_nn_and_optim.py::test_init_kaiming_normal PASSED          [ 50%]
tests/hw2/test_nn_and_optim.py::test_init_xavier_uniform PASSED          [ 75%]
tests/hw2/test_nn_and_optim.py::test_init_xavier_normal PASSED           [100%]

======================= 4 passed, 89 deselected in 0.43s =======================
```

## Question 2

### Linear

**BroadcastTo 的 gradient 问题修改**  
	维度的“右对齐”原则  
		在 NumPy 和 Needle 的广播机制中，形状是**从右向左**对齐的  
		例如：  
			输入形状 (input_shape): (5, ) (长度 `n_in=1n_in=1`)  
			输出形状 (output_shape): (10, 5) (长度 `n_out=2n_out=2`)  
			我们应该消去第一维的10，而不是第二维的5

```python
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
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        n_in = len(input_shape)
        n_out = len(output_shape)
        
        # 偏移量：比如从 (5,) 变到 (10, 5)，偏移量是 1
        offset = n_out - n_in
        axes = []
        
        for i in range(n_out):
            # 情况 1: 该维度是新增的（比如从 1维 变 2维，前面的维度都是新增的）
            if i < offset:
                axes.append(i)
            # 情况 2: 该维度原本是 1，被拉伸成了 N
            elif input_shape[i - offset] == 1:
                axes.append(i)
            
        if not axes:
            return out_grad.reshape(input_shape)
            
        # 对所有被扩展的轴求和，然后 reshape 回原始形状
        res = summation(out_grad, tuple(axes))
        return res.reshape(input_shape)
        ### END YOUR SOLUTION
```

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # 使用 Parameter 类
        # 只有被包装成 Parameter 的 Tensor，才会被 model.parameters() 收集，优化器（如 SGD）也才能更新它
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose().data)
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias:
          return out + self.bias.broadcast_to(out.shape)
        return  out
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_linear"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: anyio-4.12.1, typeguard-4.5.1, langsmith-0.7.18
collected 93 items / 85 deselected / 8 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_linear_weight_init_1 PASSED      [ 12%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_bias_init_1 PASSED        [ 25%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_forward_1 PASSED          [ 37%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_forward_2 PASSED          [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_forward_3 PASSED          [ 62%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_backward_1 PASSED         [ 75%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_backward_2 PASSED         [ 87%]
tests/hw2/test_nn_and_optim.py::test_nn_linear_backward_3 PASSED         [100%]

======================= 8 passed, 85 deselected in 0.46s =======================
```

### ReLU

**ReLU 的 gradient 问题修改**  

```python
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.where(a>0, a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 判断输入是否大于 0, 而不是输出
        # 可能存在输入 x 是一个极其微小的正数（例如 1e−20）
        # 在计算前向传播时，ReLU(x) 的结果可能因为浮点数精度问题被截断成了 0.0
        return multiply(out_grad, Tensor(node.inputs[0].realize_cached_data()>0, device=out_grad.device))
        ### END YOUR SOLUTION
```

```python
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_relu"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 91 deselected / 2 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_relu_forward_1 PASSED            [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_relu_backward_1 PASSED           [100%]

======================= 2 passed, 91 deselected in 0.43s =======================
```

### Sequential

```python
class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
          out = module(out)
        return out
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_sequential"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 91 deselected / 2 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_sequential_forward_1 PASSED      [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_sequential_backward_1 PASSED     [100%]

======================= 2 passed, 91 deselected in 0.45s =======================
```

### LogSumExp

对稳定版的 LogSumExp 公式求导，会发现所有的 max⁡(Z) 的导数项最后都会抵消掉

```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # if self.axes == None:
        #   return array_api.log(array_api.sum(array_api.exp(Z-array_api.max(Z)))) + array_api.max(Z)
        # shape = list(Z.shape)
        # flag = False
        # # 计算一个维度之后删除一个维度，同时原来坐标轴要减1
        # for axe in self.axes:
        #   if flag:
        #     axe-=1
        #   flag=True
        #   shape[axe] = 1
        #   Z = array_api.log(array_api.sum(array_api.exp(Z-array_api.max(Z, axe).reshape(shape)), axe)) + array_api.max(Z, axe)
        #   del shape[axe]

        # return Z

        # array_api（NumPy）支持一次性对多个轴求 max 和 sum
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        ret = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes, keepdims=True)) + max_z

        if self.axes is None:
          return ret.reshape(()) # 变成标量
        
        out_shape = list(Z.shape)
        axes = [self.axes] if isinstance(self.axes, int) else self.axes
        # 删除轴
        for ax in sorted(axes, reverse=True):
          out_shape.pop(ax)
        return ret.reshape(tuple(out_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # LogSumExp 的导数就是 Softmax 函数
        x = node.inputs[0]
        shape = node.inputs[0].shape

        # 求和降维, 恢复维度
        new_shape = list(shape)
        if self.axes is not None:
          # 兼容 int 和 tuple
          axes = [self.axes] if isinstance(self.axes, int) else self.axes
          for ax in axes:
            new_shape[ax] = 1
        else:
          new_shape = [1] * len(shape)

        return out_grad.reshape(new_shape).broadcast_to(shape)*exp(x-logsumexp(x, self.axes).reshape(new_shape).broadcast_to(shape))
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_op_logsumexp"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 83 deselected / 10 selected                               

tests/hw2/test_nn_and_optim.py::test_op_logsumexp_forward_1 PASSED       [ 10%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_forward_2 PASSED       [ 20%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_forward_3 PASSED       [ 30%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_forward_4 PASSED       [ 40%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_forward_5 PASSED       [ 50%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_backward_1 PASSED      [ 60%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_backward_2 PASSED      [ 70%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_backward_3 PASSED      [ 80%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_backward_5 PASSED      [ 90%]
tests/hw2/test_nn_and_optim.py::test_op_logsumexp_backward_4 PASSED      [100%]

====================== 10 passed, 83 deselected in 0.40s =======================
```

### LogSoftmax

```python
class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=1, keepdims=True)
        ret = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=1, keepdims=True)) + max_z
        
        return Z-ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # 还原softmax
        s = exp(node)

        # 还原大小
        new_shape = list(node.inputs[0].shape)
        new_shape[1] = 1 
        out = broadcast_to(reshape(summation(out_grad, 1), new_shape), node.inputs[0].shape)

        return out_grad - out*s
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_op_logsoftmax"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 90 deselected / 3 selected                                

tests/hw2/test_nn_and_optim.py::test_op_logsoftmax_forward_1 PASSED      [ 33%]
tests/hw2/test_nn_and_optim.py::test_op_logsoftmax_stable_forward_1 PASSED [ 66%]
tests/hw2/test_nn_and_optim.py::test_op_logsoftmax_backward_1 PASSED     [100%]

======================= 3 passed, 90 deselected in 0.44s =======================
```

### SoftmaxLoss

```python
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.summation(ops.add(ops.logsumexp(logits, (1,)), ops.negate(ops.summation(ops.multiply(logits, init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)), axes=(1,))))) / logits.shape[0]
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_softmax_loss"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 89 deselected / 4 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_softmax_loss_forward_1 PASSED    [ 25%]
tests/hw2/test_nn_and_optim.py::test_nn_softmax_loss_forward_2 PASSED    [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_softmax_loss_backward_1 PASSED   [ 75%]
tests/hw2/test_nn_and_optim.py::test_nn_softmax_loss_backward_2 PASSED   [100%]

======================= 4 passed, 89 deselected in 0.55s =======================
```

### LayerNorm1d

```python
class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # Parameter 包装
        # 接收了 device 和 dtype 参数，在初始化 weight 和 bias 时透传它们, 否则GPU使用报错
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ex = (x.sum((1, )) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        varx = (((x-ex) ** 2).sum((1, )) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return w * ((x-ex)/((varx+self.eps) ** 0.5)) + b
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_layernorm"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 86 deselected / 7 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_layernorm_forward_1 PASSED       [ 14%]
tests/hw2/test_nn_and_optim.py::test_nn_layernorm_forward_2 PASSED       [ 28%]
tests/hw2/test_nn_and_optim.py::test_nn_layernorm_forward_3 PASSED       [ 42%]
tests/hw2/test_nn_and_optim.py::test_nn_layernorm_backward_1 PASSED      [ 57%]
tests/hw2/test_nn_and_optim.py::test_nn_layernorm_backward_2 PASSED      [ 71%]
tests/hw2/test_nn_and_optim.py::test_nn_layernorm_backward_3 PASSED      [ 85%]
tests/hw2/test_nn_and_optim.py::test_nn_layernorm_backward_4 PASSED      [100%]

======================= 7 passed, 86 deselected in 0.44s =======================
```

### Flatten

```python
class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        nonbatch = 1
        for i in X.shape[1:]:
          nonbatch*=i
        return X.reshape((X.shape[0], int(nonbatch)))
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_flatten"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 84 deselected / 9 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_flatten_forward_1 PASSED         [ 11%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_forward_2 PASSED         [ 22%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_forward_3 PASSED         [ 33%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_forward_4 PASSED         [ 44%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_backward_1 PASSED        [ 55%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_backward_2 PASSED        [ 66%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_backward_3 PASSED        [ 77%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_backward_4 PASSED        [ 88%]
tests/hw2/test_nn_and_optim.py::test_nn_flatten_backward_5 PASSED        [100%]
```

### BatchNorm1d

```python
class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        features = x.shape[1]
        if self.training:
          ex = x.sum((0, )) / batch
          mu = ex.reshape((1, features)).broadcast_to(x.shape)
          varx = ((x-mu) ** 2).sum((0, )) / batch
          sigma = varx.reshape((1, features)).broadcast_to(x.shape)
          w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
          b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
          # 使用 .data 进行原地更新，断开与当前计算图的连接
          self.running_mean.data = (1-self.momentum)*self.running_mean.data + self.momentum*ex.data
          self.running_var.data = (1-self.momentum)*self.running_var.data + self.momentum*varx.data
          return w * ((x-mu)/((sigma+self.eps) ** 0.5)) + b
        else:
          w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
          b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
          return w * ((x-self.running_mean.reshape((1, features)).broadcast_to(x.shape))/((self.running_var.reshape((1, features)).broadcast_to(x.shape)+self.eps) ** 0.5)) + b
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_batchnorm"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 85 deselected / 8 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_check_model_eval_switches_training_flag_1 PASSED [ 12%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_forward_1 PASSED       [ 25%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_forward_affine_1 PASSED [ 37%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_backward_1 PASSED      [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_backward_affine_1 PASSED [ 62%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_running_mean_1 PASSED  [ 75%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_running_var_1 PASSED   [ 87%]
tests/hw2/test_nn_and_optim.py::test_nn_batchnorm_running_grad_1 PASSED  [100%]

======================= 8 passed, 85 deselected in 0.47s =======================
```

### Dropout

```python
class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          # 使用 *x.shape 来解包元组，这样无论 x 是几维，都能生成对应形状的掩码
          # 透传 device 以确保兼容性
          tensorb = init.randb(*x.shape, p=1-self.p, device=x.device)
          return x*(tensorb/(1-self.p))
        else:
          return x
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_dropout"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 91 deselected / 2 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_dropout_forward_1 PASSED         [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_dropout_backward_1 PASSED        [100%]

======================= 2 passed, 91 deselected in 0.51s =======================
```

### Residual

```python
class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 残差连接有一个硬性前提：self.fn(x) 输出的形状必须和输入 x 的形状完全一致
        # 通常这种情况下会引入一个“投影层”（Projection Layer），但对于 HW2 目前实现的 Residual 模块，默认输入输出形状一致是完全没问题的
        return self.fn(x)+x
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_nn_residual"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 91 deselected / 2 selected                                

tests/hw2/test_nn_and_optim.py::test_nn_residual_forward_1 PASSED        [ 50%]
tests/hw2/test_nn_and_optim.py::test_nn_residual_backward_1 PASSED       [100%]

======================= 2 passed, 91 deselected in 0.44s =======================
```

## Question 3

### SGD

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # 遍历参数更新
        for p in self.params:

          # 如果该参数没有梯度，直接跳过
          if p.grad is None:
            continue
          
          # 权重衰减通常是先作用于梯度：grad = grad + weight_decay * p
          grad = p.grad.data + self.weight_decay * p.data
          
          if p not in self.u:
            # 初始状态 u_t = 0
            self.u[p]=(1-self.momentum)*grad
          else:
            # 使用 .data 进行数值更新, 断开计算图
            self.u[p].data=self.momentum*self.u[p].data+(1-self.momentum)*grad
          
          p.data = p.data-self.lr*self.u[p].data

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_optim_sgd"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 87 deselected / 6 selected                                

tests/hw2/test_nn_and_optim.py::test_optim_sgd_vanilla_1 PASSED          [ 16%]
tests/hw2/test_nn_and_optim.py::test_optim_sgd_momentum_1 PASSED         [ 33%]
tests/hw2/test_nn_and_optim.py::test_optim_sgd_weight_decay_1 PASSED     [ 50%]
tests/hw2/test_nn_and_optim.py::test_optim_sgd_momentum_weight_decay_1 PASSED [ 66%]
tests/hw2/test_nn_and_optim.py::test_optim_sgd_layernorm_residual_1 PASSED [ 83%]
tests/hw2/test_nn_and_optim.py::test_optim_sgd_z_memory_check_1 PASSED   [100%]

======================= 6 passed, 87 deselected in 0.60s =======================
```

### Adam

```python
class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
          if p.grad is None:
            continue
          
          grad = p.grad.data + self.weight_decay*p.data

          if p not in self.m:
            self.m[p] = (1-self.beta1)*grad
          else:
            # 使用 .data 进行数值更新, 断开计算图
            self.m[p].data = self.beta1*self.m[p].data + (1-self.beta1)*grad

          if p not in self.v:
            self.v[p] = (1-self.beta2)*grad*grad
          else:
            # 使用 .data 进行数值更新, 断开计算图
            self.v[p].data = self.beta2*self.v[p].data + (1-self.beta2)*grad*grad

          bm = self.m[p] / (1-self.beta1 ** (self.t+1))
          bv = self.v[p] / (1-self.beta2 ** (self.t+1))
          p.data = p.data - self.lr*bm/(bv ** 0.5+self.eps)
        
        self.t+=1

        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_optim_adam"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 86 deselected / 7 selected                                

tests/hw2/test_nn_and_optim.py::test_optim_adam_1 PASSED                 [ 14%]
tests/hw2/test_nn_and_optim.py::test_optim_adam_weight_decay_1 PASSED    [ 28%]
tests/hw2/test_nn_and_optim.py::test_optim_adam_batchnorm_1 PASSED       [ 42%]
tests/hw2/test_nn_and_optim.py::test_optim_adam_batchnorm_eval_mode_1 PASSED [ 57%]
tests/hw2/test_nn_and_optim.py::test_optim_adam_layernorm_1 PASSED       [ 71%]
tests/hw2/test_nn_and_optim.py::test_optim_adam_weight_decay_bias_correction_1 PASSED [ 85%]
tests/hw2/test_nn_and_optim.py::test_optim_adam_z_memory_check_1 PASSED  [100%]

======================= 7 passed, 86 deselected in 0.69s =======================
```

## Question 4

#### RandomFlipHorizontal

#### RandomCrop

```python
class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          return np.flip(img, axis=1)
        else:
          return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        padimg = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0.0)
        return padimg[self.padding+shift_x:self.padding+shift_x+H, self.padding+shift_y:self.padding+shift_y+W, :]
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "flip_horizontal"
!python3 -m pytest -v -k "random_crop"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 92 deselected / 1 selected                                

tests/hw2/test_data.py::test_flip_horizontal PASSED                      [100%]

======================= 1 passed, 92 deselected in 0.37s =======================
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 92 deselected / 1 selected                                

tests/hw2/test_data.py::test_random_crop PASSED                          [100%]

======================= 1 passed, 92 deselected in 0.37s =======================
```

### Dataset

```python
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(label_filename, 'rb') as f:
          magicNumber, itemsNumber = struct.unpack('>ii', f.read(8))
          self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(image_filename, 'rb') as f:
          magicNumber, itemsNumber, rowNumber, colNumber = struct.unpack('>iiii', f.read(16))
          images = np.frombuffer(f.read(), dtype=np.uint8)
          images = images.reshape(itemsNumber, rowNumber, colNumber, 1)
          self.images = images.astype(np.float32) / 255

        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # 确保 Data Augmentation（数据增强）是在副本上进行的，不会污染原始存储在内存中的 self.images 数据集
        img = np.array(self.images[index])
        if self.transforms is not None:
          for transform in self.transforms:
            img = transform(img)
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_mnist_dataset"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 92 deselected / 1 selected                                

tests/hw2/test_data.py::test_mnist_dataset PASSED                        [100%]

======================= 1 passed, 92 deselected in 1.86s =======================
```

### Dataloader

```python
class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        if self.shuffle:
          self.ordering = np.arange(len(self.dataset))
          # 打乱整个数据集
          np.random.shuffle(self.ordering)
          self.ordering = np.array_split(self.ordering, range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
          raise StopIteration
        # Pythonic 的列表推导式, 速度稍快
        data = [self.dataset[i] for i in self.ordering[self.index]]
        self.index+=1
        result = []
        # 将样本列表 [(img1, lbl1), (img2, lbl2), ...] 转换成了类别组 [(img1, img2, ...), (lbl1, lbl2, ...)]
        for data_type_group in zip(*data):
            # 将 numpy 数组列表 stack 起来变成一个大数组
            # 沿着指定的轴将多个数组堆叠起来，从而创建一个新的多维数组
            batched_ndarray = np.stack(data_type_group)
            # 转换为 Needle Tensor 并存入结果
            result.append(Tensor(batched_ndarray))
            
        return tuple(result)
        ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_dataloader"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 91 deselected / 2 selected                                

tests/hw2/test_data.py::test_dataloader_mnist PASSED                     [ 50%]
tests/hw2/test_data.py::test_dataloader_ndarray PASSED                   [100%]

====================== 2 passed, 91 deselected in 18.70s =======================
```

## Question 5

### ResidualBlock

### MLPResNet

nn_basic 中问题修改 BatchNorm1d  
	self.momentum 是 Python 的 float（即 64 位双精度）  
	当执行 float * Tensor(float32) 时，为了保证精度不丢失，Needle 的运算符重载会将结果自动提升（Upcast）为 float64  
	但 self.running_mean 在初始化时被设定为 float32，当你试图把 float64 的计算结果塞回 float32 的容器时，autograd.py 里的 assert 就报错  

```python
class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        features = x.shape[1]
        if self.training:
          ex = x.sum((0, )) / batch
          mu = ex.reshape((1, features)).broadcast_to(x.shape)
          varx = ((x-mu) ** 2).sum((0, )) / batch
          sigma = varx.reshape((1, features)).broadcast_to(x.shape)
          w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
          b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
          # 使用 .data 进行原地更新，断开与当前计算图的连接
          new_mean = (1-self.momentum)*self.running_mean.data + self.momentum*ex.data
          self.running_mean.data = Tensor(new_mean.numpy(), dtype="float32")
          new_var = (1-self.momentum)*self.running_var.data + self.momentum*varx.data
          self.running_var.data = Tensor(new_var.numpy(), dtype="float32")
          return w * ((x-mu)/((sigma+self.eps) ** 0.5)) + b
        else:
          w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
          b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
          return w * ((x-self.running_mean.reshape((1, features)).broadcast_to(x.shape))/((self.running_var.reshape((1, features)).broadcast_to(x.shape)+self.eps) ** 0.5)) + b
        ### END YOUR SOLUTION
```

### Epoch

optim 中问题修改 Adam  
	self.lr 是 Python 的 float（即 64 位双精度）  
	当执行 float * Tensor(float32) 时，为了保证精度不丢失，Needle 的运算符重载会将结果自动提升（Upcast）为 float64  
	但 p 在初始化时被设定为 float32，当你试图把 float64 的计算结果塞回 float32 的容器时，autograd.py 里的 assert 就报错  

```python
class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
          if p.grad is None:
            continue
          
          grad = p.grad.data + self.weight_decay*p.data

          if p not in self.m:
            self.m[p] = (1-self.beta1)*grad
          else:
            # 使用 .data 进行数值更新, 断开计算图
            self.m[p].data = self.beta1*self.m[p].data + (1-self.beta1)*grad

          if p not in self.v:
            self.v[p] = (1-self.beta2)*grad*grad
          else:
            # 使用 .data 进行数值更新, 断开计算图
            self.v[p].data = self.beta2*self.v[p].data + (1-self.beta2)*grad*grad

          bm = self.m[p] / (1-self.beta1 ** (self.t+1))
          bv = self.v[p] / (1-self.beta2 ** (self.t+1))
          new_p = p.data - self.lr*bm/(bv ** 0.5+self.eps)
          p.data = ndl.Tensor(new_p.numpy(), dtype="float32")

        self.t+=1

        ### END YOUR SOLUTION
```

### Train Mnist

optim 中问题修改 SGD  
	self.lr 是 Python 的 float（即 64 位双精度）  
	当执行 float * Tensor(float32) 时，为了保证精度不丢失，Needle 的运算符重载会将结果自动提升（Upcast）为 float64  
	但 p 在初始化时被设定为 float32，当你试图把 float64 的计算结果塞回 float32 的容器时，autograd.py 里的 assert 就报错  

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # 遍历参数更新
        for p in self.params:

          # 如果该参数没有梯度，直接跳过
          if p.grad is None:
            continue
          
          # 权重衰减通常是先作用于梯度：grad = grad + weight_decay * p
          grad = p.grad.data + self.weight_decay * p.data
          
          if p not in self.u:
            # 初始状态 u_t = 0
            self.u[p]=(1-self.momentum)*grad
          else:
            # 使用 .data 进行数值更新, 断开计算图
            self.u[p].data=self.momentum*self.u[p].data+(1-self.momentum)*grad
          
          new_p = p.data-self.lr*self.u[p].data
          p.data = ndl.Tensor(new_p.numpy(), dtype="float32")

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
```

```python
def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Residual(
        nn.Sequential(
          nn.Linear(dim, hidden_dim),
          norm(hidden_dim), 
          nn.ReLU(),
          nn.Dropout(drop_prob),
          nn.Linear(hidden_dim, dim),
          norm(dim)
        )
      ),
      nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
      modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    # 使用 * 拆包列表
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
      model.train()

    total_error = 0.0
    total_loss = 0.0
    total_samples = 0
    
    # 获取损失函数实例
    loss_fn = nn.SoftmaxLoss()
    flat = nn.Flatten()

    for batch in dataloader:
      X, y = batch

      # 前向传播
      logits = model(flat(X))
      loss = loss_fn(logits, y)

      # 统计误差 (Error count)
      # logits 是 (batch, classes), y 是 (batch,)
      # 我们需要找到 logits 中每一行最大值的索引作为预测值
      preds = np.argmax(logits.numpy(), axis=1)
      total_error += (preds != y.numpy()).sum() # 这样写比 np.sum 略快

      # 统计 Loss (注意要乘以 batch size，因为 SoftmaxLoss 返回的是平均值)
      # 必须使用 .data.numpy() 或 .numpy() 来断开计算图，否则内存会爆炸
      total_loss += loss.data.numpy() * X.shape[0]
      total_samples += X.shape[0]

      # 更新参数
      if opt is not None:
        opt.reset_grad() # 清空梯度
        loss.backward() # 反向传播计算梯度
        opt.step() # 更新参数

    return total_error/total_samples, total_loss/total_samples
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 定义数据增强
    train_transforms = [
        ndl.data.RandomFlipHorizontal(),
        ndl.data.RandomCrop(padding=3)
    ]
    # 惯例：训练集需要 shuffle=True 以增加随机性，但测试集通常设为 shuffle=False
    # 虽然设为 True 不会改变准确率的结果，但在工业界和学术界，为了保证测试过程的确定性和可重复性，测试集一般不打乱
    # 训练集使用数据增强和 Shuffle, 但是使用数据增强测试不通过
    train_dataloader = ndl.data.DataLoader(
      ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")), batch_size, shuffle=True)
    # 测试集不使用增强，shuffle 设为 False
    test_dataloader = ndl.data.DataLoader(
      ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")), batch_size, shuffle=False)
    
    model = MLPResNet(784, hidden_dim=hidden_dim)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
      train_error, train_loss = epoch(train_dataloader, model, opt)
      test_error, test_loss = epoch(test_dataloader, model)
      # print 每个 epoch 的结果，方便观察模型是否收敛
        # print(f"Epoch {i}: Train Loss {train_loss:.3f}, Test Err {test_error:.3f}")
    
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION
```

```bash
!python3 -m pytest -v -k "test_mlp"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content/drive/MyDrive/10714/hw2
plugins: typeguard-4.5.1, anyio-4.13.0, langsmith-0.7.23
collected 93 items / 83 deselected / 10 selected                               

tests/hw2/test_nn_and_optim.py::test_mlp_residual_block_num_params_1 PASSED [ 10%]
tests/hw2/test_nn_and_optim.py::test_mlp_residual_block_num_params_2 PASSED [ 20%]
tests/hw2/test_nn_and_optim.py::test_mlp_residual_block_forward_1 PASSED [ 30%]
tests/hw2/test_nn_and_optim.py::test_mlp_resnet_num_params_1 PASSED      [ 40%]
tests/hw2/test_nn_and_optim.py::test_mlp_resnet_num_params_2 PASSED      [ 50%]
tests/hw2/test_nn_and_optim.py::test_mlp_resnet_forward_1 PASSED         [ 60%]
tests/hw2/test_nn_and_optim.py::test_mlp_resnet_forward_2 PASSED         [ 70%]
tests/hw2/test_nn_and_optim.py::test_mlp_train_epoch_1 PASSED            [ 80%]
tests/hw2/test_nn_and_optim.py::test_mlp_eval_epoch_1 PASSED             [ 90%]
tests/hw2/test_nn_and_optim.py::test_mlp_train_mnist_1 PASSED            [100%]

====================== 10 passed, 83 deselected in 46.86s ======================
```

