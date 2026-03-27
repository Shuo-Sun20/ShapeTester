import torch

def call_func(inputs, window_length, periodic=True, beta=12.0, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    return torch.kaiser_window(
        window_length=window_length,
        periodic=periodic,
        beta=beta,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad
    )

window_length = torch.randint(1, 20, (1,)).item()
beta = torch.rand(1).item() * 10 + 1
example_output = call_func([], window_length, beta=beta)