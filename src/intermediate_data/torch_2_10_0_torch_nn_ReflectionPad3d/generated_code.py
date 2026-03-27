import torch

def call_func(padding, inputs):
    pad_layer = torch.nn.ReflectionPad3d(padding)
    return pad_layer(inputs)

example_output = call_func(1, torch.randn(1, 1, 2, 2, 2))