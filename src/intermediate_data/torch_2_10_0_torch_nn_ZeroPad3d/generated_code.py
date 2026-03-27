import torch

def call_func(padding, inputs):
    pad_module = torch.nn.ZeroPad3d(padding)
    return pad_module(inputs)

example_output = call_func((1, 2, 3, 4, 5, 6), torch.randn(2, 3, 10, 20, 30))