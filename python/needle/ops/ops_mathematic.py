"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


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
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        base, ex = node.inputs
        return out_grad * base ** (ex-1) * ex, log(base) * (base ** ex) * out_grad
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        base = node.inputs[0]
        return out_grad * self.scalar * power_scalar(base, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        assert a.shape == b.shape
        return out_grad / b, -1 * a * (out_grad / (b ** 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        # axes is either None (swap last two) or a pair (i, j)
        self.axes = axes

    def compute(self, a):
        # Build the permutation by swapping exactly one pair of axes
        permute_axes = list(range(a.ndim))

        if self.axes is None:
            if a.ndim < 2:
                # Nothing to swap
                return a
            ax1 = a.ndim - 2
            ax2 = a.ndim - 1
        else:
            ax1, ax2 = self.axes

        permute_axes[ax1], permute_axes[ax2] = permute_axes[ax2], permute_axes[ax1]
        # IMPORTANT: do NOT overwrite self.axes here
        return a.permute(tuple(permute_axes))

    def gradient(self, out_grad, node):
        # Swapping the same pair again gives the inverse permutation.
        return transpose(out_grad, self.axes)



def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Handle PyTorch-style -1 (infer dimension)
        target = list(self.shape)
        
        if -1 in target:
            # Only one -1 allowed
            assert target.count(-1) == 1, "Only one -1 is allowed in reshape shape"
            
            # Total number of elements in input
            total_elems = 1
            for d in a.shape:
                total_elems *= d
            
            # Product of known dims in target
            known_prod = 1
            for d in target:
                if d != -1:
                    known_prod *= d
            
            assert total_elems % known_prod == 0, "Reshape -1 dimension not divisible"
            inferred = total_elems // known_prod
            
            # Replace -1 with inferred size
            idx = target.index(-1)
            target[idx] = inferred
        
        final_shape = tuple(target)
        return array_api.reshape(a.compact(), final_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        oldshape = node.inputs[0].shape
        return reshape(out_grad, oldshape)
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
        original_shape = node.inputs[0].shape
        out_shape = self.shape

        padded_shape = (1,) * (len(out_shape) - len(original_shape)) + original_shape

        axes_to_sum = [i for i, (orig_dim, new_dim) in enumerate(zip(padded_shape, out_shape)) if orig_dim == 1 and new_dim > 1]
        
        grad = out_grad
        # Sum over one axis at a time, starting from the highest axis
        for axis in reversed(sorted(axes_to_sum)):
            grad = summation(grad, axes=axis, keepdims=True)
        
        return reshape(grad, original_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Directly pass axes (None, int, or tuple) to the backend
        if self.axes is None:
            axis_to_pass = None
        elif isinstance(self.axes, tuple):
            # For single-element tuple, extract the int
            if len(self.axes) == 1:
                axis_to_pass = self.axes[0]
            else:
                # For multi-element tuple, pass the tuple directly
                axis_to_pass = self.axes
        else:
            axis_to_pass = self.axes

        # The backend should support keepdims
        result = array_api.sum(a.compact(), axis=axis_to_pass, keepdims=self.keepdims)
        
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        
        if self.axes is None:
            if self.keepdims:
                grad = out_grad
            else:
                grad_shape = (1,) * len(input_shape)
                grad = reshape(out_grad, grad_shape)
            return (broadcast_to(grad, input_shape),)

        # Single axis case
        if self.keepdims:
            return (broadcast_to(out_grad, input_shape),)
        
        # Add back the summed dimension (keepdims=False)
        axis_to_add = self.axes[0] if isinstance(self.axes, tuple) else self.axes
        
        grad_shape = list(out_grad.shape)
        grad_shape.insert(axis_to_add, 1)
        
        grad = reshape(out_grad, tuple(grad_shape))
        return (broadcast_to(grad, input_shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None, keepdims=False):
    """
    Summation that supports reducing over multiple axes.
    """
    if axes is None:
        return Summation(axes=None, keepdims=keepdims)(a)
    
    if isinstance(axes, int):
        axes = (axes,)
    
    if isinstance(axes, tuple) and len(axes) == 0:
        return a
    
    return Summation(axes=axes, keepdims=keepdims)(a)

class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        da = matmul(out_grad, transpose(b))
        db = matmul(transpose(a), out_grad)

        if da.shape != a.shape:
          da = summation(da, axes=tuple(range(len(da.shape)-len(a.shape))))
        if db.shape != b.shape:
          db = summation(db, axes=tuple(range(len(db.shape)-len(b.shape))))
        
        return da, db
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
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
        a = node.inputs[0]
        return out_grad / a
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
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = Tensor.make_const(node.realize_cached_data() > 0)
        return out_grad * mask
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        e = array_api.exp(-2 * a)
        return (1-e) / (1+e)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = node.realize_cached_data()
        dyda = 1 - y**2
        return out_grad * dyda
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, args: tuple[NDArray, ...]) -> NDArray:
        ### BEGIN YOUR SOLUTION
        input_shape = args[0].shape
        # Get the device from the input array
        device = args[0].device
        num_args = len(args)
        
        output_shape_list = list(input_shape)
        output_shape_list.insert(self.axis, num_args)
        output_shape = tuple(output_shape_list)
        
        out = device.empty(output_shape)
        
        for i, arr in enumerate(args):
            # 3a. Expand arr's dims to match the slice
            # e.g., (A, B) -> (A, 1, B) if axis=1
            
            shape_with_new_dim = (1,) + arr.shape
            arr_b = arr.broadcast_to(shape_with_new_dim)
            
            n_dims = arr.ndim
            permute_list = list(range(1, n_dims + 1))
            permute_list.insert(self.axis, 0)
            arr_expanded = arr_b.permute(tuple(permute_list))
            
            slicer = [slice(None)]*out.ndim
            slicer[self.axis] = i
            
            out[tuple(slicer)] = arr_expanded
            
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor]:
        ### BEGIN YOUR SOLUTION
        return (split(out_grad, self.axis),)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, A: NDArray) -> tuple[NDArray, ...]:
        ### BEGIN YOUR SOLUTION
        # This implementation is correct and uses only NDArray primitives
        num_splits = A.shape[self.axis]
        
        target_shape_list = list(A.shape)
        del target_shape_list[self.axis]
        target_shape = tuple(target_shape_list)
        
        outputs = []
        for i in range(num_splits):
            slicer = [slice(None)] * A.ndim
            slicer[self.axis] = i
            
            # Get the (A, 1, B) view
            split_i_view = A[tuple(slicer)]
            
            # "Squeeze" by compacting and reshaping to (A, B)
            split_i_squeezed = split_i_view.compact().reshape(target_shape)
            outputs.append(split_i_squeezed)
            
        return tuple(outputs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: tuple[Tensor, ...], node: Tensor) -> tuple[Tensor]:
        ### BEGIN YOUR SOLUTION
        # This implementation is correct
        return (stack(out_grad, self.axis),)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = [slice(None)] * a.ndim
        step = self.dilation + 1
        
        # Create a set for efficient lookup
        axes_set = set(self.axes) 

        for i in range(a.ndim):
            if i in axes_set:
                # The new dimension is old_dim * (dilation + 1)
                new_shape[i] = a.shape[i] * step
                # The slice to assign to is every 'step'-th element
                slices[i] = slice(None, None, step)
        
        # Create the new, zero-filled output array
        out = a.device.full(tuple(new_shape), 0.0, dtype=a.dtype)
        
        # Use __setitem__ with the strided slices to place 'a's data
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # The gradient of scattering (Dilate) is gathering (UnDilate)
        return UnDilate(self.axes, self.dilation)(out_grad)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)] * a.ndim
        step = self.dilation + 1
        axes_set = set(self.axes)

        for i in range(a.ndim):
            if i in axes_set:
                # The slice to gather from is every 'step'-th element
                slices[i] = slice(None, None, step)
        
        # Use __getitem__ with the strided slices to gather data
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # The gradient of gathering (UnDilate) is scattering (Dilate)
        return Dilate(self.axes, self.dilation)(out_grad)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        p = self.padding
        s = self.stride

        # 1. Apply Padding to Input Tensor A (NHWC)
        if p > 0:
            # axes = ( (N_pad_left, N_pad_right), (H_pad_left, H_pad_right), ... )
            # We pad H (axis 1) and W (axis 2)
            padding_axes = ((0, 0), (p, p), (p, p), (0, 0))
            A_pad = A.pad(padding_axes)
        else:
            A_pad = A

        # Get dimensions for computation
        N, H_pad, W_pad, C_in = A_pad.shape
        K, K_w, K_in, C_out = B.shape  # B is (K, K, C_in, C_out)
        
        # Assertions to ensure dimensions match
        assert K == K_w, "Kernel must be square"
        assert C_in == K_in, "Input channels of A must match input channels of B"

        # 2. Calculate Output Dimensions
        # O = (I_pad - K) / S + 1
        H_out = (H_pad - K) // s + 1
        W_out = (W_pad - K) // s + 1

        # 3. Create im2col view using as_strided
        # Get the strides of the padded input
        Ns, Hs, Ws, Cs = A_pad.strides
        
        # Shape of the strided view
        im2col_shape = (N, H_out, W_out, K, K, C_in)
        
        # Strides for the strided view
        im2col_strides = (Ns, Hs * s, Ws * s, Hs, Ws, Cs)
        
        # Use the NDArray.as_strided method
        A_strided = A_pad.as_strided(
            shape=im2col_shape,
            strides=im2col_strides
        )

        # 4. Reshape for MatMul
        # Flatten patches: (N, H_out, W_out, K, K, C_in) -> (N*H_out*W_out, K*K*C_in)
        inner_dim = K * K * C_in
        # NDArray.reshape does not support -1, so we calculate the full shape
        A_flat = A_strided.compact().reshape((N * H_out * W_out, inner_dim))
        
        # Flatten weights: (K, K, C_in, C_out) -> (K*K*C_in, C_out)
        B_flat = B.compact().reshape((inner_dim, C_out))
        
        # 5. Perform MatMul
        # (N*H_out*W_out, K*K*C_in) @ (K*K*C_in, C_out) -> (N*H_out*W_out, C_out)
        out_flat = A_flat @ B_flat
        
        # 6. Reshape output
        # (N*H_out*W_out, C_out) -> (N, H_out, W_out, C_out)
        return out_flat.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        
        # Get kernel size (K), stride (s), and padding (p) from forward pass
        K = B.shape[0]
        s = self.stride
        p = self.padding

        # 1. --- Calculate Gradient w.r.t. Input A (grad_A) ---
        # Formula: grad_A = conv(dilate(out_grad), flip(swap_channels(B)))
        
        # Swap C_in and C_out channels: (K, K, C_in, C_out) -> (K, K, C_out, C_in)
        B_swapped = transpose(B, axes=(2, 3))
        # Flip kernel B's spatial dims: (K, K, C_out, C_in)
        B_flipped = flip(B_swapped, axes=(0, 1))

        # Dilate out_grad if stride > 1
        out_grad_dilated = out_grad
        if s > 1:
            # Dilate spatial dimensions (H and W, which are axes 1 and 2)
            out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=s - 1)

        # Calculate padding for the "transpose convolution"
        # The formula (from slides) is K - 1 - p_original
        p_new = K - 1 - p
        
        # The new convolution is always stride 1
        grad_A = conv(out_grad_dilated, B_flipped, stride=1, padding=p_new)


        # 2. --- Calculate Gradient w.r.t. Weights B (grad_B) ---
        # Formula: grad_B = conv(permute(A), permute(dilate(out_grad)))
        
        # Permute A: (N, H, W, C_in) -> (C_in, H, W, N)
        # This treats C_in as the batch dim and N as the channel dim
        A_perm = transpose(A, axes=(0, 3))
        
        # Dilate out_grad (same as for grad_A)
        out_grad_dilated = out_grad
        if s > 1:
            out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=s - 1)

        # Permute out_grad_dilated: (N, H_d, W_d, C_out) -> (H_d, W_d, N, C_out)
        # This makes it a kernel: (K_h, K_w, C_in_kernel, C_out_kernel)
        # We need two transposes to permute (0, 1, 2, 3) -> (1, 2, 0, 3)
        # (N, H_d, W_d, C_out) -> (H_d, N, W_d, C_out)
        out_grad_perm_temp = transpose(out_grad_dilated, (0, 1))
        # (H_d, N, W_d, C_out) -> (H_d, W_d, N, C_out)
        out_grad_perm = transpose(out_grad_perm_temp, (1, 2))

        # Convolve A_perm with out_grad_perm
        # Use the original padding 'p'
        # Stride is 1 (dilation already handled the effect of the forward stride)
        grad_B_perm = conv(A_perm, out_grad_perm, stride=1, padding=p)
        
        # Output is (C_in, K, K, C_out).
        # We need (K, K, C_in, C_out).
        # Permute: (0, 1, 2, 3) -> (1, 2, 0, 3)
        # (C_in, K, K, C_out) -> (K, C_in, K, C_out)
        grad_B_temp = transpose(grad_B_perm, (0, 1))
        # (K, C_in, K, C_out) -> (K, K, C_in, C_out)
        grad_B = transpose(grad_B_temp, (1, 2))

        return grad_A, grad_B
        ### END YOUR SOLUTION

def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


