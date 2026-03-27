import torch

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return torch.cosh(input=input_tensor, out=out)

# Generate random input tensor
torch.manual_seed(42)
inputs = torch.randn(4)
example_output = call_func(inputs)