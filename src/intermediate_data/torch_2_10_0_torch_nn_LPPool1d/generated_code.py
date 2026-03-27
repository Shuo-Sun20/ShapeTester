import torch
import torch.nn as nn

def call_func(norm_type, kernel_size, stride=None, ceil_mode=False, inputs=None):
    if stride is None:
        stride = kernel_size
    module = nn.LPPool1d(norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)
    return module(inputs)

# Generate random input tensor
torch.manual_seed(42)
inputs = torch.randn(20, 16, 50)

# Call the function
example_output = call_func(2, 3, stride=2, inputs=inputs)