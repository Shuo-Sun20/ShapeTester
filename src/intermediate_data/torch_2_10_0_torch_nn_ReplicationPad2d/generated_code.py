import torch

def call_func(padding, inputs):
    layer = torch.nn.ReplicationPad2d(padding)
    output = layer(inputs)
    return output

input_tensor = torch.randn(2, 3, 5, 5)
example_output = call_func((1, 1, 2, 0), input_tensor)