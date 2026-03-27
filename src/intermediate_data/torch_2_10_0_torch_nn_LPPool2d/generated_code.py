import torch
import torch.nn as nn

def call_func(norm_type, kernel_size, stride=None, ceil_mode=False, inputs=None):
    if stride is None:
        stride = kernel_size
    pool_layer = nn.LPPool2d(norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)
    output = pool_layer(inputs)
    return output

input_tensor = torch.randn(20, 16, 50, 32)
example_output = call_func(2, 3, stride=2, inputs=input_tensor)