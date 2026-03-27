import torch

def call_func(inputs, upper=False, out=None):
    B, L = inputs
    return torch.cholesky_solve(B, L, upper=upper, out=out)

torch.manual_seed(42)
n, k = 3, 2
# Generate a symmetric positive-definite matrix A
A = torch.randn(n, n, dtype=torch.float64)
A = A @ A.T + torch.eye(n, dtype=torch.float64) * 1e-3
# Compute its Cholesky decomposition (lower triangular by default)
L = torch.linalg.cholesky(A)
# Generate a random right-hand side matrix B
B = torch.randn(n, k, dtype=torch.float64)
# Call the function
example_output = call_func(inputs=[B, L])