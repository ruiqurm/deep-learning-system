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
        # for p in self.params:
         for idx, param in enumerate(self.params):
            if param.grad is None:
                continue
            # TODO: change this to a more efficient way
            graident = ndl.Tensor(param.grad.detach().cached_data,dtype=param.dtype)+self.weight_decay*ndl.Tensor(param.detach().cached_data,dtype=param.dtype)
            self.u[param] = self.momentum * self.u.get(param, 0) + (1 - self.momentum)*graident
            self.params[idx].data -= self.lr * self.u[param]
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
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            grad = ndl.Tensor(param.grad.detach().cached_data,dtype=param.dtype)+self.weight_decay*param.data
            
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad.data
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad.data * grad.data)
            # bias correction
            m_hat = self.m[param].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[param].data / (1 - self.beta2 ** self.t)
            
            param.data -= self.lr * m_hat / (v_hat**0.5+self.eps)     
        ### END YOUR SOLUTION
