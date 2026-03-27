import torch

def call_func(inputs, size=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None):
    ccol_indices, row_indices, values = inputs
    if size is not None:
        return torch.sparse_bsc_tensor(
            ccol_indices,
            row_indices,
            values,
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
            check_invariants=check_invariants
        )
    else:
        return torch.sparse_bsc_tensor(
            ccol_indices,
            row_indices,
            values,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
            check_invariants=check_invariants
        )

torch.manual_seed(42)
ccol_indices = torch.tensor([0, 2, 3], dtype=torch.int64)
row_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
values = torch.randn(3, 2, 2)  
example_output = call_func([ccol_indices, row_indices, values], size=(4, 4))