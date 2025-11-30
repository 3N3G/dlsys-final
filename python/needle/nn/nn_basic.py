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
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
          self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)))

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        bias_broadcasted_shape = list(X.shape)
        bias_broadcasted_shape[-1] = self.out_features
        if self.bias is None:
          return ops.matmul(X, self.weight)
        return ops.add(ops.matmul(X, self.weight), ops.broadcast_to(self.bias, tuple(bias_broadcasted_shape)))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flattened_size = 1
        for i in range(1, len(X.shape)):
            flattened_size *= X.shape[i]
            
        return ops.reshape(X, (batch_size, flattened_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class GELU(Module):
    """
    Approximate GELU:
        0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    """
    def __init__(self):
        super().__init__()
        # sqrt(2/pi)
        self.c = 0.7978845608028654  

    def forward(self, x):
        return 0.5 * x * (
            1.0 + ops.tanh(self.c * (x + 0.044715 * (x ** 3)))
        )

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for m in self.modules:
          y = m.forward(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        y_one_hot = init.one_hot(num_classes, y, device=logits.device)
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        loss_per_sample = log_sum_exp - z_y

        return ops.summation(loss_per_sample) / batch_size
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
        batch_size = x.shape[0]
        if self.training:
          E = ops.summation(x, axes=(0,))
          E = ops.divide_scalar(E, batch_size)
          bigE = ops.reshape(E, (1, self.dim))
          bigE = ops.broadcast_to(bigE, x.shape)
          diffs = ops.add(x, ops.mul_scalar(bigE, -1))
          V = ops.power_scalar(diffs, 2)
          V = ops.summation(V, axes=(0,))
          V = ops.divide_scalar(V, batch_size)
          
          self.running_mean = ops.add(
            ops.mul_scalar(self.running_mean, 1 - self.momentum),
            ops.mul_scalar(E, self.momentum)
          ).data
          self.running_var = ops.add(
            ops.mul_scalar(self.running_var, 1 - self.momentum),
            ops.mul_scalar(V, self.momentum)
          ).data

          V_broadcast = ops.reshape(V, (1, self.dim))
          V_broadcast = ops.broadcast_to(V_broadcast, x.shape)
          std = ops.power_scalar(ops.add_scalar(V_broadcast, self.eps), 0.5)
          z = ops.divide(diffs, std)
        else:
          E_broadcast = ops.reshape(self.running_mean, (1, self.dim))
          E_broadcast = ops.broadcast_to(E_broadcast, x.shape)
          diffs = ops.add(x, ops.mul_scalar(E_broadcast, -1))
          
          V_broadcast = ops.reshape(self.running_var, (1, self.dim))
          V_broadcast = ops.broadcast_to(V_broadcast, x.shape)
          std = ops.power_scalar(ops.add_scalar(V_broadcast, self.eps), 0.5)
          z = ops.divide(diffs, std)

        w = ops.reshape(self.weight, (1, self.dim))
        w = ops.broadcast_to(w, x.shape)
        b = ops.reshape(self.bias, (1, self.dim))
        b = ops.broadcast_to(b, x.shape)
        
        return ops.add(ops.multiply(w, z), b)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

# class BatchNorm2d(BatchNorm1d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x: Tensor):
#         # Expect NCHW
#         s = x.shape
#         assert len(s) == 4, f"BatchNorm2d expects 4D input (N,C,H,W), got {s}"
#         N, C, H, W = s

#         # NCHW -> NHWC
#         x_nhwc = x.transpose((1, 2)).transpose((2, 3))  # (N, H, W, C)

#         # Flatten spatial + batch dims, keep channels = C
#         # Shape becomes (N * H * W, C), but we let reshape infer N*H*W
#         x_flat = x_nhwc.reshape((-1, C))

#         # Apply 1D batchnorm on the last dim (C)
#         y_flat = super().forward(x_flat)  # (N*H*W, C)

#         # Unflatten back to (N, H, W, C)
#         y_nhwc = y_flat.reshape((N, H, W, C))

#         # NHWC -> NCHW
#         y_nchw = y_nhwc.transpose((2, 3)).transpose((1, 2))
#         return y_nchw


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        # Expect NCHW
        s = x.shape
        assert len(s) == 4, f"MaxPool2d expects 4D input (N,C,H,W), got {s}"
        N, C, H, W = s

        # NCHW -> NHWC
        x_nhwc = x.transpose((1, 2)).transpose((2, 3))   # (N, H, W, C)

        # Call backend max_pool
        y = ops.max_pool(x_nhwc, self.kernel_size, self.stride)
        ys = y.shape

        # --- Fix: handle the extra dimension your current max_pool is adding ---
        # We are currently seeing (N, 1, H_out, W_out, C)
        if len(ys) == 5 and ys[1] == 1:
            N2, one, H_out, W_out, C2 = ys
            assert N2 == N and one == 1 and C2 == C, f"Unexpected max_pool shape {ys}"
            # Drop the dummy dim: (N, 1, H_out, W_out, C) -> (N, H_out, W_out, C)
            y = y.reshape((N, H_out, W_out, C))
            ys = y.shape

        # Otherwise we expect NHWC
        if len(ys) != 4:
            raise ValueError(f"MaxPool2d: unexpected shape from ops.max_pool: {ys}")

        # NHWC -> NCHW
        y_nchw = y.transpose((2, 3)).transpose((1, 2))   # (N, C, H_out, W_out)
        return y_nchw



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5,
                 device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones((dim,), device=device, dtype=dtype))
        self.bias = Parameter(init.zeros((dim,), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        x has shape (..., dim) where the *last* dimension is the feature dim.
        Normalize along that last dimension.
        """
        # Last axis index: for 2D (N, D) -> 1; for 3D (B, T, D) -> 2
        axis = len(x.shape) - 1

        # ---- mean over last dim ----
        # shape: (..., 1)
        mean = ops.summation(x, axis, keepdims=True)
        mean = ops.divide_scalar(mean, self.dim)

        # broadcast mean to full x.shape
        mean_full = ops.broadcast_to(mean, x.shape)

        # centered: x - mean
        diffs = ops.add(x, ops.mul_scalar(mean_full, -1.0))

        # ---- variance over last dim ----
        sq = ops.power_scalar(diffs, 2)                  # (..., dim)
        var = ops.summation(sq, axis, keepdims=True)     # (..., 1)
        var = ops.divide_scalar(var, self.dim)
        var = ops.add_scalar(var, self.eps)

        # std, broadcasted
        std = ops.power_scalar(var, 0.5)                 # (..., 1)
        std_full = ops.broadcast_to(std, x.shape)        # (..., dim)

        # normalized
        z = ops.divide(diffs, std_full)                  # (..., dim)

        # ---- affine transform: gamma, beta in shape (dim,) ----
        w = ops.broadcast_to(self.weight, x.shape)       # (dim,) -> (..., dim)
        b = ops.broadcast_to(self.bias, x.shape)         # (dim,) -> (..., dim)

        return ops.add(ops.multiply(w, z), b)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # No dropout when not training or p = 0
        if (not self.training) or self.p == 0.0:
            return x

        keep_p = 1.0 - self.p

        # randb default dtype is bool, so override to float32
        # and make sure the mask lives on the same device as x.
        mask = init.randb(
            *x.shape,
            p=keep_p,
            device=x.device,
            dtype="float32", 
            requires_grad=False,
        )

        # Inverted dropout: scale by 1 / keep_p
        out = ops.multiply(x, mask)
        out = ops.divide_scalar(out, keep_p)
        return out
        ### END YOUR SOLUTION



class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
