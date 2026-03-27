import torch

def call_func(inputs, alpha=1, out=None):
    input_tensor, other_tensor = inputs[0], inputs[1]
    return torch.sub(input_tensor, other_tensor, alpha=alpha, out=out)

example_output = call_func([torch.randn(3, 4), torch.randn(3, 4)], alpha=2)