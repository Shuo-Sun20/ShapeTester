import torch

def call_func(padding, inputs):
    pad_layer = torch.nn.ReflectionPad2d(padding)
    output = pad_layer(inputs)
    return output

example_output = call_func(padding=2, inputs=torch.randn(1, 1, 3, 3))