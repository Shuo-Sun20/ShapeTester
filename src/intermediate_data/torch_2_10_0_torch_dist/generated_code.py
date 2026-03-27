import torch

def call_func(inputs, p=2):
    input_tensor = inputs[0]
    other_tensor = inputs[1]
    return torch.dist(input_tensor, other_tensor, p)

x = torch.randn(4)
y = torch.randn(4)
example_output = call_func([x, y], p=2)