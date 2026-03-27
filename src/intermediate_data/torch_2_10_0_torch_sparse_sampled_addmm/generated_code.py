import torch

def call_func(inputs, beta=1.0, alpha=1.0, out=None):
    input_tensor, mat1, mat2 = inputs
    return torch.sparse.sampled_addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha, out=out)

m, n, k = 4, 5, 3
input_sparse = torch.randn(m, n).to_sparse_csr()
mat1 = torch.randn(m, k)
mat2 = torch.randn(k, n)
inputs = [input_sparse, mat1, mat2]
example_output = call_func(inputs)