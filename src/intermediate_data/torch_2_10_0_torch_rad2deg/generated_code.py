import torch

def call_func(inputs, out=None):
    # Since torch.rad2deg is a function with single input tensor, 
    # we extract it from the list (if provided as list) or use directly
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
        
    return torch.rad2deg(input=input_tensor, out=out)

# Create random input tensor
inputs = torch.rand(3, 2) * 6.283  # Random values between 0 and 2π
example_output = call_func(inputs)