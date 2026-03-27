import torch

def call_func(padding, inputs):
    pad_layer = torch.nn.CircularPad1d(padding)
    output = pad_layer(inputs)
    return output

example_input = torch.randn(2, 8)
example_output = call_func((1, 2), example_input)