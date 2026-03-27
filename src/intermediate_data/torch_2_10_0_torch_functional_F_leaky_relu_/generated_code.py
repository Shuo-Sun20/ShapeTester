import torch
import torch.nn.functional as F

def call_func(inputs, negative_slope=0.01):
    return torch.nn.functional.leaky_relu_(inputs, negative_slope)

example_input = torch.randn(3, 4)
example_output = call_func(example_input, negative_slope=0.1)