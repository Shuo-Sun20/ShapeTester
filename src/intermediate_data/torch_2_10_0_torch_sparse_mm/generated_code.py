import torch

def call_func(inputs, reduce="sum"):
    mat1, mat2 = inputs
    if mat1.layout in [torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc]:
        return torch.sparse.mm(mat1, mat2, reduce=reduce)
    else:
        return torch.sparse.mm(mat1, mat2)

sparse_mat = torch.randn(3, 4).to_sparse_csr()
dense_mat = torch.randn(4, 2)
example_output = call_func([sparse_mat, dense_mat])