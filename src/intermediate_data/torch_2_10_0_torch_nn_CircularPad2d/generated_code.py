import torch
from torch.nn import CircularPad2d

def call_func(padding, inputs):
    pad_layer = CircularPad2d(padding)
    return pad_layer(inputs)

input_tensor = torch.randn(1, 3, 32, 32)
example_output = call_func(2, input_tensor)