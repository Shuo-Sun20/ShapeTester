import torch

def call_func(padding, value, inputs):
    pad_layer = torch.nn.ConstantPad2d(padding=padding, value=value)
    output = pad_layer(inputs)
    return output

input_tensor = torch.randn(1, 2, 2)
example_output = call_func(padding=2, value=3.5, inputs=input_tensor)