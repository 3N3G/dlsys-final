"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.use_bias = bias
        
        # Calculate "same" padding. P = (K-1)/2
        # This assumes stride=1 for "same" padding calculation
        self.padding = (self.kernel_size - 1) // 2
        
        # Calculate fan_in and fan_out for initializations
        # fan_in = num_input_features * kernel_height * kernel_width
        fan_in = self.in_channels * (self.kernel_size ** 2)
        fan_out = self.out_channels * (self.kernel_size ** 2)
        
        # Initialize weight (K, K, C_in, C_out)
        weight_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out, shape=weight_shape, 
                                                    device=device, dtype=dtype))
        
        # Initialize bias (C_out,)
        if self.use_bias:
            # Use uniform init: [-bound, bound] where bound = 1 / sqrt(fan_in)
            bound = 1.0 / (fan_in ** 0.5)
            self.bias = Parameter(init.rand(self.out_channels, low=-bound, high=bound, 
                                            device=device, dtype=dtype))
        else:
            self.bias = None 
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
      # Input x is NCHW (N, C, H, W)
      
      # 1. Convert NCHW -> NHWC
      x_nhwc = x.transpose((1, 2)).transpose((2, 3))
      
      # 2. Apply convolution
      out_nhwc = ops.conv(x_nhwc, self.weight, self.stride, self.padding)
      
      # 3. Add bias if enabled
      if self.use_bias:
          # Let broadcasting happen automatically during addition
          bias_reshaped = self.bias.reshape((1, 1, 1, self.out_channels))
          bias_shape = (1, 1, 1, self.out_channels)
          out_nhwc = out_nhwc + self.bias.reshape(bias_shape).broadcast_to(out_nhwc.shape)
          
      # 4. Convert NHWC -> NCHW
      out_nchw = out_nhwc.transpose((2, 3)).transpose((1, 2))
      
      return out_nchw