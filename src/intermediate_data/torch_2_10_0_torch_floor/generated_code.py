import torch

def call_func(inputs, out=None):
    return torch.floor(inputs[0], out=out)

# Generate random input tensor
torch.manual_seed(0)
input_tensor = torch.randn(3, 4) * 5
example_output = call_func([input_tensor])