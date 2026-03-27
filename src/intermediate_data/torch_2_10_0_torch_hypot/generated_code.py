import torch

def call_func(inputs, out=None):
    return torch.hypot(inputs[0], inputs[1], out=out)

# Generate random tensors for input
tensor1 = torch.randn(3, 1)
tensor2 = torch.randn(1, 4)
inputs = [tensor1, tensor2]

# Call the function and save output
example_output = call_func(inputs)