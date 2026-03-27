import torch

def call_func(inputs, size, dtype, layout, device, pin_memory, requires_grad, check_invariants):
    compressed_indices, plain_indices, values = inputs
    return torch.sparse_compressed_tensor(
        compressed_indices,
        plain_indices,
        values,
        size=size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        check_invariants=check_invariants
    )

batch_size = 2
nrows = 3
ncols = 4
blocksize = (2, 2)
dense_dim = 2
compressed_dim_size = nrows

compressed_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int64).repeat(batch_size, 1)
plain_indices = torch.tensor([0, 2, 1, 3, 0, 3], dtype=torch.int64).repeat(batch_size, 1)
values = torch.randn(batch_size, compressed_dim_size, blocksize[0], blocksize[1], dense_dim)
size = (batch_size, nrows * blocksize[0], ncols * blocksize[1], dense_dim)

example_output = call_func(
    inputs=[compressed_indices, plain_indices, values],
    size=size,
    dtype=torch.float32,
    layout=torch.sparse_bsr,
    device="cpu",
    pin_memory=False,
    requires_grad=False,
    check_invariants=False
)