import torch

def call_func(inputs, upper=False, out=None):
    return torch.linalg.cholesky(inputs[0], upper=upper, out=out)

n = 4
A = torch.randn(n, n, dtype=torch.float64)
A = A @ A.mT + torch.eye(n, dtype=torch.float64)
example_output = call_func([A])