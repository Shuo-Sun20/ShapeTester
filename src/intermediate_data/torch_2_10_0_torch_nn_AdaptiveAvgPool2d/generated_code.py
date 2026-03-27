import torch
import torch.nn as nn

def call_func(output_size, inputs):
    layer = nn.AdaptiveAvgPool2d(output_size)
    output = layer(inputs)
    return output

example_input = torch.randn(1, 64, 8, 9)
example_output = call_func((5, 7), example_input)