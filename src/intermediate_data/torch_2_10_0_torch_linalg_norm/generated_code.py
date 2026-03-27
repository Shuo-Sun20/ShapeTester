import torch

def call_func(inputs, ord=None, dim=None, keepdim=False, out=None, dtype=None):
    A = inputs[0] if isinstance(inputs, list) else inputs
    return torch.linalg.norm(A, ord=ord, dim=dim, keepdim=keepdim, out=out, dtype=dtype)

example_tensor = torch.randn(3, 4)
example_output = call_func(inputs=[example_tensor], ord='fro')