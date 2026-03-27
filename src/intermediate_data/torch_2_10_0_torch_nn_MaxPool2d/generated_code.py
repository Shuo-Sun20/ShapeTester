import torch
import torch.nn as nn

def call_func(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, inputs=None):
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, 
                         dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    return pool(inputs)

input_tensor = torch.randn(20, 16, 50, 32)
example_output = call_func(kernel_size=3, stride=2, inputs=input_tensor)