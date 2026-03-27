import torch

def call_func(inputs, hermitian=False, out=None):
    LD, pivots, B = inputs
    return torch.linalg.ldl_solve(LD, pivots, B, hermitian=hermitian, out=out)

# Generate random symmetric matrix A
A = torch.randn(3, 3, dtype=torch.float64)
A = A @ A.T  # Make symmetric
LD, pivots, info = torch.linalg.ldl_factor_ex(A)
B = torch.randn(3, 2, dtype=torch.float64)
example_output = call_func([LD, pivots, B])