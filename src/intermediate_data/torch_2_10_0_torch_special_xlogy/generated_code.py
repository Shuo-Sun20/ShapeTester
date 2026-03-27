import torch

def call_func(inputs, out=None):
    return torch.special.xlogy(inputs[0], inputs[1], out=out)

x = torch.randn(3, 4)
y = torch.rand(3, 4) + 0.5  # Ensure positive for valid log
example_output = call_func([x, y])