"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


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


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        nonbatch = 1
        for i in X.shape[1:]:
          nonbatch*=i
        return X.reshape((X.shape[0], int(nonbatch)))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

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


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.summation(ops.add(ops.logsumexp(logits, (1,)), ops.negate(ops.summation(ops.multiply(logits, init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)), axes=(1,))))) / logits.shape[0]
        ### END YOUR SOLUTION


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
