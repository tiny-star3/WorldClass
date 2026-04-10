"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


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
