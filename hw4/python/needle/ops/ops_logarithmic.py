from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    # FOR 2D NDArray AND AXIS=1
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # 1. Get max with keepdims (shape will be (N, 1))
        m = array_api.max(Z, axis=1, keepdims=True)
        
        # 2. Broadcast m to Z's shape (N, C)
        m_broadcast = array_api.broadcast_to(m, Z.shape)
        
        # 3. Compute sum(exp(Z-m_broadcast)) (shape will be (N, 1))
        sum_exp = array_api.sum(array_api.exp(Z - m_broadcast), axis=1, keepdims=True)
        
        # 4. log(sum_exp) + m (shape will be (N, 1))
        logsumexp_val = array_api.log(sum_exp) + m
        
        # 5. Broadcast (N, 1) logsumexp_val to (N, C) for final subtraction
        logsumexp_val_broadcast = array_api.broadcast_to(logsumexp_val, Z.shape)
        
        return Z - logsumexp_val_broadcast
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node) # from ops_mathematic
        grad_sum = summation(out_grad, axes=(1,))
        grad_sum = reshape(grad_sum, (grad_sum.shape[0], 1))
        
        # Broadcast grad_sum to match Z's shape
        grad_sum_broadcast = broadcast_to(grad_sum, softmax.shape)
        
        # Gradient: out_grad - softmax * sum(out_grad)
        return out_grad - softmax * grad_sum_broadcast
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # 1. Get max, but keep dims so we can broadcast
        m = array_api.max(Z, axis=self.axes, keepdims=True)
        
        # 2. Manually broadcast m to Z's shape
        m_broadcast = array_api.broadcast_to(m, Z.shape)
        
        # 3. Subtract the broadcasted max
        exp_term = array_api.exp(Z - m_broadcast)
        
        # 4. Sum and log
        log_sum_exp = array_api.log(array_api.sum(exp_term, axis=self.axes, keepdims=True))
        
        # 5. Add back the original (un-broadcasted) max
        nZ = log_sum_exp + m
        
        if self.axes is not None:
            axes_tuple = (self.axes,) if isinstance(self.axes, int) else tuple(self.axes)
            nZ = array_api.squeeze(nZ, axis=axes_tuple)
        else:
            nZ = array_api.squeeze(nZ)

        if nZ.shape == () or (hasattr(nZ, 'size') and nZ.size == 1 and nZ.ndim == 0):
            return nZ

        return nZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        axes = self.axes
        
        out_grad_expanded = out_grad
        if axes is not None:
            axes_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
            # Allow for negative axes
            axes_normalized = tuple(ax if ax >= 0 else len(Z.shape) + ax for ax in axes_tuple)
            new_shape = list(out_grad.shape)
            for ax in sorted(axes_normalized):
                new_shape.insert(ax, 1)
            out_grad_expanded = reshape(out_grad, tuple(new_shape))
        
        logsumexp_expanded = reshape(node, out_grad_expanded.shape)
        
        # Softmax = exp(Z - logsumexp(Z))
        softmax = exp(Z - broadcast_to(logsumexp_expanded, Z.shape))
        
        # Broadcast out_grad to match Z's shape
        out_grad_broadcast = broadcast_to(out_grad_expanded, Z.shape)
        return out_grad_broadcast * softmax


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)