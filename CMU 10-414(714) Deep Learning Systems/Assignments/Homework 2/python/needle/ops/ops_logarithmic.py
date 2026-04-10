from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

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


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


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


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)