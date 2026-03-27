import torch

def call_func(inputs, q, dim=None, keepdim=False, interpolation='linear', out=None):
    return torch.quantile(inputs[0], q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)

example_output = call_func(
    inputs=[torch.randn(2, 3)],
    q=torch.tensor([0.25, 0.5, 0.75]),
    dim=1,
    keepdim=True,
    interpolation='linear'
)