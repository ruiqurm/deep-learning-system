"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle.init import kaiming_uniform, ones, zeros
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
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


def _child_modules(value: object) -> List["Module"]:
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
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(kaiming_uniform(out_features,1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        """
        shape(X) = (N, in_features)
        shape(self.weight) = (in_features, out_features)
        shape(self.bias) = (1, out_features)
        shape(output) = (N, out_features)
        """
        result = X @ self.weight
        if self.bias:
            if self.bias.shape[0] != 1:
                right_shape = [1 for _ in range(len(result.shape))]
                right_shape[-1] = self.bias.shape[-1]
                self.bias = self.bias.reshape(right_shape)
            bias = ops.broadcast_to(self.bias, result.shape)
            return result + bias
        else:
            return result
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch = X.shape[0]
        # other_dim = 1
        # for i in range(1, len(X.shape)):
        #     other_dim *= X.shape[i]
        return X.reshape((batch,-1 ))
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_classes = logits.shape[-1]
        y_onehot = init.one_hot(num_classes,y)
        zy = (logits * y_onehot).sum(axes=(1,))
        loss = (ops.logsumexp(logits, (1,)) - zy).sum() / logits.shape[0]
        return loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(zeros(dim, device=device, dtype=dtype))
        self.running_mean = zeros(dim, device=device, dtype=dtype)
        self.running_var = ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n,m = x.shape # n stands for batch size, m stands for features
        weight = ops.broadcast_to(self.weight.reshape((1,m)),x.shape)
        bias = ops.broadcast_to(self.bias.reshape((1,m)),x.shape)
        if self.training:
            current_mean = ops.summation(x,axes=(0,)) / n
            current_mean_broadcast = ops.broadcast_to(current_mean.reshape((1,m)),x.shape)
            current_var = ops.summation((x - current_mean_broadcast) ** 2,axes=(0,)) / n
            self.running_mean = (1-self.momentum) * self.running_mean.data + self.momentum * current_mean.data
            self.running_var  = (1-self.momentum) * self.running_var.data + self.momentum * current_var.data
            mean = current_mean_broadcast
            var = ops.broadcast_to(current_var.reshape((1,m)),x.shape)
            return weight * (x - mean)/((var+self.eps)**0.5) + bias
        else:
            mean = ops.broadcast_to(self.running_mean.reshape((1,m)),x.shape)
            var = ops.broadcast_to(self.running_var.reshape((1,m)),x.shape)
            return weight * (x - mean)/((var+self.eps)**0.5) + bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n,m = x.shape # n stands for batch size, m stands for features
        mean = ops.broadcast_to( (ops.summation(x,axes=(1,)) / m).reshape((n,1)),x.shape)
        var = ops.broadcast_to((ops.summation((x - mean) ** 2,axes=(1,)) / m).reshape((n,1)),x.shape)
        weight = ops.broadcast_to(self.weight.reshape((1,m)),x.shape)
        bias = ops.broadcast_to(self.bias.reshape((1,m)),x.shape)
        return weight * (x - mean)/((var+self.eps)**0.5) + bias
        
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x*init.randb(*x.shape,p=1-self.p) / (1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

