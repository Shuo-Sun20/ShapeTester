import torch

def call_func(padding, inputs):
    pad_layer = torch.nn.ZeroPad1d(padding)
    output = pad_layer(inputs)
    return output

example_input = torch.randn(1, 2, 4)
example_output = call_func(2, example_input)