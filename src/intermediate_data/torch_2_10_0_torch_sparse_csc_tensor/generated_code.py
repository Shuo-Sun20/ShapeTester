import torch

def call_func(inputs, size=None, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None):
    ccol_indices, row_indices, values = inputs
    return torch.sparse_csc_tensor(ccol_indices, row_indices, values, size, dtype=dtype, device=device, pin_memory=pin_memory, requires_grad=requires_grad, check_invariants=check_invariants)

ccol_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
row_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
values = torch.tensor([1.0, 2.0, 3.0, 4.0])
inputs = [ccol_indices, row_indices, values]
example_output = call_func(inputs, size=(2, 2), dtype=torch.float64, requires_grad=False)