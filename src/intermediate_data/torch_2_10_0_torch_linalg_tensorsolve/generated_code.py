import torch

def call_func(inputs, dims=None, out=None):
    A, B = inputs[0], inputs[1]
    return torch.linalg.tensorsolve(A, B, dims=dims, out=out)

A = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
B = torch.randn(2 * 3, 4)
example_output = call_func([A, B])