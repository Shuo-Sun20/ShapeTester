import torch

def call_func(inputs, left=True, adjoint=False, out=None):
    LU, pivots, B = inputs
    return torch.linalg.lu_solve(LU, pivots, B, left=left, adjoint=adjoint, out=out)

torch.manual_seed(0)
A = torch.randn(3, 3)
LU, pivots = torch.linalg.lu_factor(A)
B = torch.randn(3, 2)
example_output = call_func([LU, pivots, B])