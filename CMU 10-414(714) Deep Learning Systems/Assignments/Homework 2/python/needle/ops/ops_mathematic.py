"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


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


def relu(a):
    return ReLU()(a)

