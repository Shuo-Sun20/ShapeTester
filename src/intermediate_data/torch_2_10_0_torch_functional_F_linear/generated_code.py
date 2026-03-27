import torch
import torch.nn.functional as F

def call_func(inputs, weight, bias=None):
    input_tensor = inputs
    return F.linear(input_tensor, weight, bias)

example_input = torch.randn(4, 6)
example_weight = torch.randn(5, 6)
example_bias = torch.randn(5)

example_output = call_func(example_input, example_weight, bias=example_bias)