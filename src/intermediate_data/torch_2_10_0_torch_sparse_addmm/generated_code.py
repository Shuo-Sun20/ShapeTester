import torch

def call_func(inputs, beta=1., alpha=1.):
    mat, mat1, mat2 = inputs
    return torch.sparse.addmm(mat, mat1, mat2, beta=beta, alpha=alpha)

# Create a sparse COO matrix (mat1)
sparse_indices = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.int64)
sparse_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
sparse_shape = torch.Size([3, 3])
sparse_mat = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_shape)

# Create dense matrices (mat and mat2)
dense_mat = torch.randn(3, 3, dtype=torch.float32)
dense_mat2 = torch.randn(3, 3, dtype=torch.float32)

inputs = [dense_mat, sparse_mat, dense_mat2]
example_output = call_func(inputs, beta=1.0, alpha=1.0)