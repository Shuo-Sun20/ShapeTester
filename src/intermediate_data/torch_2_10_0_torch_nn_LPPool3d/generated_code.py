import torch
import torch.nn as nn

def call_func(norm_type, kernel_size, stride=None, ceil_mode=False, inputs=None):
    if stride is None:
        pool_layer = nn.LPPool3d(norm_type, kernel_size, ceil_mode=ceil_mode)
    else:
        pool_layer = nn.LPPool3d(norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)
    
    if isinstance(inputs, list):
        output = pool_layer(*inputs)
    else:
        output = pool_layer(inputs)
    
    return output

input_tensor = torch.randn(20, 16, 50, 44, 31)
example_output = call_func(2, 3, stride=2, inputs=input_tensor)