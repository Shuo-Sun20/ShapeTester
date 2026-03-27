import torch

def call_func(inputs, dim, dtype=None):
    return torch.special.softmax(inputs, dim=dim, dtype=dtype)

input_tensor = torch.randn(2, 3, 4)
example_output = call_func(inputs=input_tensor, dim=1)