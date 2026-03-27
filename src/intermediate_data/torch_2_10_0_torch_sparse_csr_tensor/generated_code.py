import torch

def call_func(inputs, size=None, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None):
    crow_indices, col_indices, values = inputs
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size, dtype=dtype, device=device, pin_memory=pin_memory, requires_grad=requires_grad, check_invariants=check_invariants)

crow_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
values = torch.randn(4)
example_output = call_func([crow_indices, col_indices, values], size=(2, 2))