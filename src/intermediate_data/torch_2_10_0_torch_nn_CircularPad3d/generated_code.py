import torch

def call_func(padding, inputs):
    pad_layer = torch.nn.CircularPad3d(padding)
    output = pad_layer(inputs)
    return output

input_tensor = torch.randn(16, 3, 8, 320, 480)
example_output = call_func(3, input_tensor)