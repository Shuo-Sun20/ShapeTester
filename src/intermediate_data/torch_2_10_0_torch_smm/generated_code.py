import torch

def call_func(inputs):
    sparse_matrix, dense_matrix = inputs
    return torch.smm(sparse_matrix, dense_matrix)

# Generate random sparse and dense tensors
sparse_indices = torch.tensor([[0, 1, 2], [0, 2, 3]])
sparse_values = torch.randn(3)
sparse_matrix = torch.sparse_coo_tensor(sparse_indices, sparse_values, (3, 4))
dense_matrix = torch.randn(4, 2)

# Call function and save output
example_output = call_func([sparse_matrix, dense_matrix])