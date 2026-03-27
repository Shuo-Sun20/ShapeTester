import torch

def call_func(inputs, size=None, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None, is_coalesced=None):
    indices, values = inputs[0], inputs[1]
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=size,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        check_invariants=check_invariants,
        is_coalesced=is_coalesced
    )

indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
values = torch.randn(3)
example_output = call_func([indices, values], size=[3, 3])