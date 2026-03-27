import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    return torch.trunc(input=input_tensor, out=out)

# Generate random input tensor
input_tensor = torch.randn(4)
inputs = [input_tensor]
example_output = call_func(inputs)