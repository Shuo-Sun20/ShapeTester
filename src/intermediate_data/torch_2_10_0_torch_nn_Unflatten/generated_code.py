import torch
import torch.nn as nn

def call_func(dim, unflattened_size, inputs):
    unflatten_instance = nn.Unflatten(dim, unflattened_size)
    output = unflatten_instance(inputs)
    return output

example_input = torch.randn(2, 50)
example_output = call_func(1, (2, 5, 5), example_input)