import torch

def call_func(inputs, out=None):
    return torch.linalg.inv(A=inputs, out=out)

A = torch.randn(4, 4)
A = A @ A.T + torch.eye(4) * 1e-3
example_output = call_func(inputs=A)